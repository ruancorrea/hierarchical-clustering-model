"""Quickstart Demo for the Hierarchical Clustering Model (HCM).

Demonstrates the 3-phase execution pipeline of HCM:
1. Offline Historical Training (H1: Spatial Outlier Filtering + Spatial Clustering)
2. Offline Parameter Calibration (H2: Optimal k1 and e-rate tuning)
3. Online Real-Time Routing (H3: 2-Level Hierarchical Assignment & Intra-cluster TSP)
"""

import logging
from pathlib import Path
from models import CVRPInstance
from hcm import HCMPolicy, HCMParams
from application import create_policy
from core.engine import OnlineCVRPEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Quickstart")


def main():
    base_dir = Path(__file__).resolve().parent
    train_dir = base_dir / "instances" / "train" / "pa-0"
    batch_file = base_dir / "instances" / "batches" / "pa-0" / "batch_0.json"

    if not train_dir.exists() or not batch_file.exists():
        logger.error(f"Sample instances not found at {train_dir} or {batch_file}")
        return

    # 1. Load Training and Operational Instances
    logger.info("Loading training instances...")
    train_files = sorted(train_dir.glob("*.json"))[:20]  # First 20 historical days
    train_instances = [CVRPInstance.from_file(f) for f in train_files]
    logger.info(f"Loaded {len(train_instances)} historical training instances.")

    logger.info(f"Loading operational batch: {batch_file.name}")
    eval_instance = CVRPInstance.from_file(batch_file)
    logger.info(f"Target instance: {eval_instance.name} ({len(eval_instance.deliveries)} deliveries, capacity={eval_instance.vehicle_capacity})")

    # 2. Configure HCM Hyperparameters
    params = HCMParams(
        n_clusters=8,             # Number of regional spatial macro-clusters (k1)
        e_rate=0.0,               # Outlier contamination rate (0.0 for dense, 0.02 for dispersed)
        n_unit_loads=28,          # Number of physical Unit Loads (sorting bays)
        alpha_criteria=0.94,      # Cluster capacity fill threshold before route split
        time_limit_ms_tsp=1000,   # TSP solver time limit per cluster (ms)
        seed=42,
    )

    # 3. Instantiate and Train HCM Policy (H1 + H2)
    logger.info("Initializing HCM Policy...")
    policy = create_policy("hcm", params)

    logger.info("Training HCM on historical delivery demand (Offline Phase H1+H2)...")
    policy.train(train_instances)
    logger.info("Offline training completed successfully.")

    # 4. Online Real-Time Routing (H3)
    logger.info("Executing Online Routing Engine (H3)...")
    engine = OnlineCVRPEngine(
        policy=policy,
        n_unit_loads=params.n_unit_loads,
        alpha_criteria=params.alpha_criteria,
        time_limit_ms_tsp=params.time_limit_ms_tsp,
        show_progress=True,
    )

    solution, total_distance = engine.run(eval_instance)

    # 5. Output Summary
    print("\n" + "=" * 60)
    print("           HCM ROUTING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Instance Name       : {eval_instance.name}")
    print(f"Deliveries Served   : {len(eval_instance.deliveries)}")
    print(f"Total Routes Created: {len(solution.vehicles)}")
    print(f"Total Distance (km) : {total_distance:.2f} km")
    print(f"Avg Packages/Route  : {len(eval_instance.deliveries) / len(solution.vehicles):.2f}")
    print("=" * 60)

    # Save output solution
    output_path = base_dir / "solution_demo.json"
    solution.to_file(output_path)
    logger.info(f"Full vehicle routing solution saved to {output_path}")


if __name__ == "__main__":
    main()
