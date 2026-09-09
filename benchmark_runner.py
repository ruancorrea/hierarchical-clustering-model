"""Benchmark Runner for Multi-Seed & Hyperparameter Search Experiments.

Executes extensive experiments across:
- Multiple random seeds (e.g., 10 to 30 seeds for statistical reliability)
- Configurable parameter search grids from JSON (e.g., custom n_clusters ranges, e_rates including 0.0)
- Multiple methods (HCM, KMeans, Sweep)
- Multiple benchmark instances (Batches or Dev sets)

Generates:
- master_results.csv (all runs consolidated)
- benchmark_report.md (complete statistical report with Mean ± Std, CV%, Friedman & Wilcoxon tests)
- benchmark_tables.tex (ready-to-use LaTeX tables for papers and dissertation)
"""

import argparse
import csv
import json
import logging
import os
import sys

# Prevent OpenMP/BLAS deadlocks across multiprocessing loops
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from dataclasses import asdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "models.py").exists() else Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from distances import preload_distance_cache, get_active_distance_metric
from hcm import HCMParams, HCM_OFFLINE_V2, HCMParamsV2
from models import CVRPInstance

from application import create_policy, get_data, init_worker, solve_task
from stats import StatisticalAnalyzer, df_to_markdown_table, df_to_latex_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkRunner")


def run_benchmark(
    train_path: str,
    eval_path: str,
    methods: List[str],
    seeds: List[int],
    params_path: Optional[str] = None,
    output_dir: str = "solutions/benchmarks",
) -> Path:
    """Run full multi-seed benchmark experiment."""
    train_instances, eval_files = get_data(train_path, eval_path)

    # Detect region
    region = train_instances[0].region if train_instances and hasattr(train_instances[0], "region") else "pa-0"
    if not region or region == "pa-0":
        for part in Path(train_path).parts:
            if "-" in part and any(part.startswith(prefix) for prefix in ["pa", "df", "rj"]):
                region = part
                break

    # Preload memory-mapped distance cache once in main process
    logger.info(f"Ensuring distance cache is initialized for region {region}...")
    preload_distance_cache(region=region)

    # Load base params
    search_json = Path(params_path) if params_path else Path(__file__).resolve().parent / "params" / "search.json"
    if search_json.is_file():
        logger.info(f"Loading base parameters from {search_json}")
        with open(search_json) as f:
            raw_params = json.load(f)
    else:
        logger.info("Using default parameter grid...")
        raw_params = {
            "n_clusters": list(range(3, 29)),
            "e_rate": [0.0, 0.001, 0.005, 0.01],
            "n_unit_loads": 28,
            "alpha_criteria": 0.94,
            "test_size": 0.05,
            "time_limit_ms_tsp": 1000,
        }

    # Setup experiment folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(output_dir) / f"experiment_{timestamp}_{region}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    master_records = []
    total_runs = len(methods) * len(seeds)
    run_idx = 0

    logger.info(f"Starting Benchmark Experiment: {len(methods)} methods × {len(seeds)} seeds = {total_runs} runs.")
    logger.info(f"Evaluation instances: {len(eval_files)} files from {eval_path}")

    for seed in seeds:
        for method in methods:
            run_idx += 1
            method_upper = method.upper()
            logger.info(f"--- [Run {run_idx}/{total_runs}] Method: {method_upper} | Seed: {seed} ---")

            # Setup params for this seed (always fresh search/default params)
            current_params_dict = dict(raw_params)
            current_params_dict["seed"] = seed
            current_params_dict["method"] = method.lower()

            # Instantiate HCMParams
            params_obj = HCMParams(**{k: v for k, v in current_params_dict.items() if k in HCMParams.__dataclass_fields__})

            # Create policy & fit (Offline Phase)
            policy = create_policy(method, params_obj)
            start_offline = time.perf_counter()
            policy.fit(train_instances, n_unit_loads=params_obj.n_unit_loads)
            time_offline = time.perf_counter() - start_offline

            # Update fitted params
            if hasattr(policy, "n_clusters") and isinstance(policy.n_clusters, int):
                current_params_dict["n_clusters"] = policy.n_clusters
            elif hasattr(policy, "clustering") and policy.clustering is not None:
                current_params_dict["n_clusters"] = policy.clustering.n_clusters
            if hasattr(policy, "e_rate") and isinstance(policy.e_rate, (int, float)):
                current_params_dict["e_rate"] = float(policy.e_rate)

            logger.info(
                f"Fitted {method_upper} (seed={seed}, region={region}): "
                f"n_clusters={current_params_dict.get('n_clusters')}, e_rate={current_params_dict.get('e_rate')}, "
                f"offline_time={time_offline:.2f}s"
            )

            # Sub-directory for this specific run
            run_sub_dir = exp_dir / f"{method.lower()}_seed_{seed}"
            details_dir = run_sub_dir / "details"
            details_dir.mkdir(parents=True, exist_ok=True)

            current_params_dict["region"] = region
            current_params_dict["distance_metric"] = get_active_distance_metric(region=region)

            with open(run_sub_dir / "params.json", "w", encoding="utf-8") as f:
                json.dump(current_params_dict, f, indent=2)

            # Online Phase
            tasks = [
                (
                    file_path,
                    policy,
                    params_obj.n_unit_loads,
                    params_obj.alpha_criteria,
                    params_obj.time_limit_ms_tsp,
                    details_dir,
                    region,
                    time_offline,
                )
                for file_path in eval_files
            ]

            n_workers = min(os.cpu_count() or 4, len(eval_files))
            with Pool(n_workers, initializer=init_worker, initargs=(region,), maxtasksperchild=1) as pool:
                run_results = list(tqdm(pool.imap(solve_task, tasks), total=len(tasks), desc=f"{method_upper} (seed={seed})"))
                pool.close()
                pool.join()
            
            import gc
            del tasks
            gc.collect()

            # Record into master results
            for res in run_results:
                rec = {
                    "instance": res["instance"],
                    "method": method_upper,
                    "seed": seed,
                    "distance": res["distance"],
                    "time_online": res["time_online"],
                    "time_offline": res["time_offline"],
                    "routes": res["routes"],
                }
                master_records.append(rec)

            # Save per-run results.csv
            run_csv = run_sub_dir / "results.csv"
            pd.DataFrame(run_results).to_csv(run_csv, index=False)

    # Save master_results.csv
    master_df = pd.DataFrame(master_records)
    master_csv = exp_dir / "master_results.csv"
    master_df.to_csv(master_csv, index=False)
    logger.info(f"Saved consolidated master results to {master_csv}")

    # Generate Statistical Analysis Report
    _generate_benchmark_report(master_df, exp_dir, methods, seeds, len(eval_files))
    return exp_dir


def _generate_benchmark_report(
    master_df: pd.DataFrame, exp_dir: Path, methods: List[str], seeds: List[int], n_instances: int
):
    """Generate multi-seed statistical analysis and robustness summary."""
    report_lines = []
    report_lines.append("# 🔬 Relatório do Experimento Científico Multi-Seed\n")
    report_lines.append(f"- **Data de Execução:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report_lines.append(f"- **Total de Instâncias Avaliadas:** {n_instances}")
    report_lines.append(f"- **Métodos Comparados:** {', '.join([m.upper() for m in methods])}")
    report_lines.append(f"- **Seeds Testados ($S={len(seeds)}$):** {seeds}\n")

    # 1. Tabela Detalhada de Desempenho por Batch (Média ± Desvio sobre os Seeds)
    report_lines.append("## 1. Desempenho por Instância / Batch (Média ± Desvio Padrão sobre os Seeds)")
    per_batch_rows = []
    unique_methods = sorted(list(set(master_df["method"].unique())))

    for inst, grp in master_df.groupby("instance"):
        row = {"Instance": inst}
        for m in unique_methods:
            sub = grp[grp["method"] == m]["distance"]
            if not sub.empty:
                if len(sub) > 1 and not np.isnan(sub.std()):
                    row[m] = f"{sub.mean():.2f} ± {sub.std():.2f}"
                else:
                    row[m] = f"{sub.mean():.2f}"
            else:
                row[m] = "N/A"
        per_batch_rows.append(row)

    per_batch_df = pd.DataFrame(per_batch_rows)
    report_lines.append(df_to_markdown_table(per_batch_df))
    report_lines.append("")

    # 2. Tabela de Robustez Geral dos Métodos
    report_lines.append("## 2. Tabela de Robustez Geral dos Métodos (Consolidado sobre todos os Seeds)")
    grouped = master_df.groupby("method")
    robustness_rows = []
    best_mean_dist = grouped["distance"].mean().min()

    for m_name, group in grouped:
        dist_mean = group["distance"].mean()
        dist_std = group["distance"].std() if len(group["distance"]) > 1 and not np.isnan(group["distance"].std()) else 0.0
        dist_med = group["distance"].median()
        dist_min = group["distance"].min()
        dist_max = group["distance"].max()
        cv_pct = (dist_std / dist_mean * 100.0) if dist_mean > 0 else 0.0
        gap_pct = ((dist_mean - best_mean_dist) / best_mean_dist * 100.0) if best_mean_dist > 0 else 0.0

        time_on_mean = group["time_online"].mean()
        time_off_mean = group["time_offline"].mean()
        routes_mean = group["routes"].mean()

        dist_label = f"{dist_mean:.2f} ± {dist_std:.2f}" if dist_std > 0 else f"{dist_mean:.2f}"

        robustness_rows.append({
            "Method": m_name,
            "Distance (Mean ± Std)": dist_label,
            "Median": f"{dist_med:.2f}",
            "CV (%)": f"{cv_pct:.2f}%",
            "Min": f"{dist_min:.2f}",
            "Max": f"{dist_max:.2f}",
            "Gap (%)": f"{gap_pct:.2f}%",
            "Time Online (s)": f"{time_on_mean:.2f}",
            "Time Offline (s)": f"{time_off_mean:.2f}",
            "Avg Routes": f"{routes_mean:.1f}",
        })

    robustness_df = pd.DataFrame(robustness_rows)
    report_lines.append(df_to_markdown_table(robustness_df))
    report_lines.append("")

    # 3. Análise Estatística Agregada por Instância (Média de cada método por batch)
    avg_per_instance = master_df.groupby(["instance", "method"])["distance"].mean().reset_index()
    analyzer = StatisticalAnalyzer(avg_per_instance, metric="distance")

    report_lines.append("## 3. Testes de Hipótese Estatística (Comparação Pareada sobre as Médias dos Batches)")
    report_lines.append(analyzer.summary_report())

    # Write report markdown
    report_path = exp_dir / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Write LaTeX tables
    latex_path = exp_dir / "benchmark_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% --- Tabela Detalhada por Batch ---\n")
        f.write(
            df_to_latex_table(
                per_batch_df,
                caption="Distância média e desvio padrão obtidos em cada batch através dos múltiplos seeds.",
                label="tab:per_batch_results",
            )
        )
        f.write("\n\n% --- Tabela de Robustez Geral Multi-Seed ---\n")
        f.write(
            df_to_latex_table(
                robustness_df,
                caption="Desempenho e Robustez Geral dos Métodos sob múltiplos seeds aleatórios.",
                label="tab:multi_seed_robustness",
            )
        )
        f.write("\n\n% --- Tabela de Testes Estatísticos Post-Hoc ---\n")
        f.write(analyzer.generate_latex_tables())

    logger.info(f"Saved benchmark report to {report_path}")
    logger.info(f"Saved LaTeX tables to {latex_path}")
    print("\n" + "=" * 70)
    print("\n".join(report_lines))


def parse_seeds_arg(seeds_input: Optional[List[Union[str, int]]]) -> List[int]:
    """Parse seeds list, supporting integers, lists of numbers, and range strings like '0..9' or '0-9'."""
    if not seeds_input:
        return [42, 101, 203, 314, 425, 536, 647, 758, 869, 970]

    parsed = []
    for item in seeds_input:
        s_str = str(item).strip()
        if ".." in s_str:
            start_s, end_s = s_str.split("..")
            parsed.extend(list(range(int(start_s), int(end_s) + 1)))
        elif "-" in s_str and not s_str.startswith("-"):
            start_s, end_s = s_str.split("-")
            parsed.extend(list(range(int(start_s), int(end_s) + 1)))
        else:
            parsed.append(int(s_str))
    return sorted(list(set(parsed)))


def recalculate_report_from_dir(exp_dir_path: Union[str, Path]):
    """Recalculate and display the multi-seed benchmark report from an existing experiment directory."""
    exp_dir = Path(exp_dir_path)
    master_csv = exp_dir / "master_results.csv" if exp_dir.is_dir() else exp_dir
    if not master_csv.is_file():
        raise FileNotFoundError(f"master_results.csv not found in {exp_dir_path}")

    master_df = pd.read_csv(master_csv)
    target_dir = master_csv.parent
    methods = sorted(list(master_df["method"].unique()))
    seeds = sorted(list(master_df["seed"].unique())) if "seed" in master_df.columns else [0]
    n_instances = len(master_df["instance"].unique())

    _generate_benchmark_report(master_df, target_dir, methods, seeds, n_instances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Benchmark Runner for Online CVRP")
    parser.add_argument("--report", type=str, help="Path to an existing experiment folder or master_results.csv to recalculate report")
    parser.add_argument("--train_instances", type=str, help="Path to training instances")
    parser.add_argument("--eval_instances", type=str, help="Path to evaluation instances")
    parser.add_argument("--methods", nargs="+", default=["hcm", "kmeans", "sweep", "greedy_tsp"], help="Methods to benchmark")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0..9"],
        help="List of random seeds (e.g. 0 1 2 3 4 5 6 7 8 9, 0..9, or {0..9})",
    )
    parser.add_argument("--params", type=str, help="Path to JSON file with parameters or search intervals")
    parser.add_argument("--output", type=str, default="solutions/benchmarks", help="Output directory")

    args = parser.parse_args()

    if args.report:
        recalculate_report_from_dir(args.report)
    else:
        if not args.train_instances or not args.eval_instances:
            parser.error("--train_instances and --eval_instances are required when running a new benchmark.")
        seeds_list = parse_seeds_arg(args.seeds)
        run_benchmark(
            train_path=args.train_instances,
            eval_path=args.eval_instances,
            methods=args.methods,
            seeds=seeds_list,
            params_path=args.params,
            output_dir=args.output,
        )
