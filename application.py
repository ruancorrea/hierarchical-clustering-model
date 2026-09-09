"""CLI Application Entry Point for Hierarchical Cluster Model (HCM) and Baselines."""

import csv
import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import CVRPInstance
from distances import preload_distance_cache, get_active_distance_metric
from hcm import HCMPolicy, HCMParams

from core.engine import OnlineCVRPEngine
from core.base_policy import BaseRoutingPolicy
from policies.kmeans_policy import KMeansPolicy
from policies.sweep_policy import SweepPolicy
from policies.greedy_tsp_policy import GreedyTSPPolicy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_data(train_instances_arg: str, eval_instances_arg: str) -> Tuple[List[CVRPInstance], List[Path]]:
    """Load training instances and testing instance file paths deterministically."""
    train_path = Path(train_instances_arg)
    train_files = [train_path] if train_path.is_file() else sorted([f for f in train_path.iterdir() if f.is_file() and f.suffix == ".json"])
    train_instances = [CVRPInstance.from_file(f) for f in train_files[:240]]

    eval_path = Path(eval_instances_arg)
    eval_files = [eval_path] if eval_path.is_file() else sorted([f for f in eval_path.iterdir() if f.is_file() and f.suffix == ".json"])

    return train_instances, eval_files


def init_worker(region: Optional[str] = None):
    """Worker process initializer attaching to the zero-copy memory-mapped distance cache."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    preload_distance_cache(region=region)


def solve_task(task_args) -> dict:
    """Worker task solving a single CVRP evaluation instance."""
    (
        file_path,
        policy,
        n_unit_loads,
        alpha_criteria,
        time_limit_ms_tsp,
        details_dir,
        region,
        time_offline,
    ) = task_args

    instance = CVRPInstance.from_file(file_path)
    engine = OnlineCVRPEngine(
        policy=policy,
        n_unit_loads=n_unit_loads,
        alpha_criteria=alpha_criteria,
        time_limit_ms_tsp=time_limit_ms_tsp,
        show_progress=False,
    )

    start_online = time.perf_counter()
    solution, distance = engine.run(instance)
    time_online = time.perf_counter() - start_online
    num_routes = len(solution.vehicles)

    if details_dir:
        solution.to_file(Path(details_dir) / f"{instance.name}.json")

    print(
        f"[{policy.name}] region: {region} | instance: {instance.name} | "
        f"distance: {distance:.4f} km | routes: {num_routes} | time_online: {time_online:.2f}s"
    )

    return {
        "instance": instance.name,
        "method": policy.name,
        "distance": round(distance, 4),
        "time_online": round(time_online, 4),
        "time_offline": round(time_offline, 4),
        "routes": num_routes,
    }


def create_policy(method: str, params: HCMParams) -> BaseRoutingPolicy:
    """Instantiate the routing policy corresponding to the requested method."""
    method_lower = method.lower()
    if method_lower == "hcm":
        return HCMPolicy(
            n_clusters=params.n_clusters,
            e_rate=params.e_rate,
            n_unit_loads=params.n_unit_loads,
            seed=params.seed,
        )
    elif method_lower in ["kmeans", "k-means", "kmeans_greedy", "kmeans-greedy"]:
        return KMeansPolicy(seed=params.seed)
    elif method_lower in ["sweep", "qro_sweep", "qrp_sweep", "qro-sweep"]:
        return SweepPolicy()
    elif method_lower in ["greedy_tsp", "greedy-tsp", "greedy"]:
        return GreedyTSPPolicy()
    else:
        raise ValueError(f"Unknown solution method: {method}. Choose from ['hcm', 'kmeans', 'sweep', 'greedy_tsp'].")


if __name__ == "__main__":
    parser = ArgumentParser(description="HCM: Hierarchical Cluster Model for Online CVRP with Physical Unit Loads")

    parser.add_argument("--train_instances", type=str, required=True, help="Path to training instances directory/file")
    parser.add_argument("--eval_instances", type=str, required=True, help="Path to evaluation instances directory/file")
    parser.add_argument(
        "--method",
        type=str,
        default="hcm",
        choices=["hcm", "kmeans", "kmeans_greedy", "sweep", "qro_sweep", "greedy_tsp", "greedy"],
        help="Solution method to use (hcm, kmeans, sweep, greedy_tsp)",
    )
    parser.add_argument("--output", type=str, default="solutions", help="Base output directory")
    parser.add_argument("--params", type=str, help="Path to JSON file with parameters")
    parser.add_argument("--n_clusters", type=int, default=None, help="Fixed n_clusters (k1)")
    parser.add_argument("--e_rate", type=float, default=None, help="Fixed outlier contamination rate e")
    parser.add_argument("--n_unit_loads", type=int, default=28, help="Number of physical Unit Loads")
    parser.add_argument("--alpha_criteria", type=float, default=0.94, help="Cluster capacity threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_instances, eval_files = get_data(args.train_instances, args.eval_instances)

    if args.params:
        logger.info(f"Loading parameters from {args.params}")
        params = HCMParams.from_file(args.params)
    else:
        search_json = Path(__file__).resolve().parent / "params" / "search.json"
        if search_json.is_file() and args.n_clusters is None and args.e_rate is None:
            logger.info(f"Loading default search parameters from {search_json}")
            params = HCMParams.from_file(search_json)
            params.seed = args.seed
        else:
            params = HCMParams(
                n_unit_loads=args.n_unit_loads,
                n_clusters=args.n_clusters if args.n_clusters is not None else list(range(3, 29)),
                e_rate=args.e_rate if args.e_rate is not None else [0.0, 0.001, 0.005, 0.01],
                alpha_criteria=args.alpha_criteria,
                seed=args.seed,
            )

    # Detect region
    region = train_instances[0].region if train_instances and hasattr(train_instances[0], "region") else "pa-0"
    if not region or region == "pa-0":
        for part in Path(args.train_instances).parts:
            if "-" in part and any(part.startswith(prefix) for prefix in ["pa", "df", "rj"]):
                region = part
                break

    # Preload distance cache once in main process
    preload_distance_cache(region=region)

    # OFFLINE PHASE: Train solution method
    policy = create_policy(args.method, params)
    logger.info(f"Starting OFFLINE phase for method: {policy.name}...")
    start_offline = time.perf_counter()
    policy.fit(train_instances, n_unit_loads=params.n_unit_loads)
    time_offline = time.perf_counter() - start_offline
    logger.info(f"OFFLINE phase completed in {time_offline:.2f}s")

    # Output directory
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = Path(args.output or "solutions")
    run_dir = base_output_dir / f"{now_str}_{region}"
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    # Save params.json
    params_dict = asdict(params)
    params_dict["method"] = args.method
    if hasattr(policy, "n_clusters") and isinstance(policy.n_clusters, int):
        params_dict["n_clusters"] = policy.n_clusters
    elif hasattr(policy, "clustering") and policy.clustering is not None:
        params_dict["n_clusters"] = policy.clustering.n_clusters
    if hasattr(policy, "e_rate") and isinstance(policy.e_rate, (int, float)):
        params_dict["e_rate"] = float(policy.e_rate)
    params_dict["region"] = region
    params_dict["distance_metric"] = get_active_distance_metric(region=region)

    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params_dict, f, indent=2)
    logger.info(f"Run output directory: {run_dir}")

    # ONLINE PHASE: Parallel evaluation
    tasks = [
        (
            file_path,
            policy,
            params.n_unit_loads,
            params.alpha_criteria,
            params.time_limit_ms_tsp,
            details_dir,
            region,
            time_offline,
        )
        for file_path in eval_files
    ]

    results = []
    with Pool(min(os.cpu_count() or 4, len(tasks)), initializer=init_worker, initargs=(region,)) as pool:
        results = list(tqdm(pool.imap(solve_task, tasks), total=len(tasks), desc="Online Solving Pool"))
        pool.close()
        pool.join()

    # Save consolidated CSV
    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance", "method", "distance", "time_online", "time_offline", "routes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"Saved results summary to {csv_path}")
    logger.info(f"Saved {len(results)} solution JSON files to {details_dir}")
