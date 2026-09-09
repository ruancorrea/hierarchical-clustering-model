"""Core Offline Training & Hyperparameter Tuning for Hierarchical Cluster Model (HCM)."""

import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models import CVRPInstance, Delivery, JSONDataclassMixin
from hcm.offline.outliers import remove_outliers_isolation_forest
from hcm.offline.allocation import allocate_unit_loads
from hcm.offline.evaluator import evaluate_hcm_configuration

logger = logging.getLogger(__name__)


def _init_offline_worker():
    """Worker process initializer disabling multithreading in BLAS/OpenMP."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


@dataclass
class HCMParams(JSONDataclassMixin):
    n_clusters: Union[int, List[int]] = 8
    e_rate: Union[float, List[float]] = 0.0
    n_unit_loads: int = 28
    alpha_criteria: float = 0.94
    test_size: float = 0.05
    time_limit_ms_tsp: int = 1_000
    seed: int = 42
    use_minibatch: bool = True


# Aliases for compatibility
HCMParamsV2 = HCMParams


class HCM_OFFLINE:
    """Offline 2-Level Hierarchical Model Builder and Parameter Calibration."""

    def __init__(
        self,
        data: List[CVRPInstance],
        params: Optional[HCMParams] = None,
        n_clusters: Optional[Union[int, List[int], range]] = None,
        e_rate: Optional[Union[float, List[float]]] = None,
        n_unit_loads: Optional[int] = None,
        alpha_criteria: Optional[float] = None,
        test_size: Optional[float] = None,
        seed: Optional[int] = None,
        use_minibatch: bool = True,
        **kwargs,
    ):
        self.params = params or HCMParams()
        self.data = data
        self.instance_basic_data = data[0]

        self.H = [pkg for instance in data for pkg in instance.deliveries]
        self.n_clusters = n_clusters if n_clusters is not None else self.params.n_clusters
        self.e_rate = e_rate if e_rate is not None else self.params.e_rate
        self.n_unit_loads = n_unit_loads if n_unit_loads is not None else self.params.n_unit_loads
        self.alpha_criteria = (
            alpha_criteria if alpha_criteria is not None else self.params.alpha_criteria
        )
        self.test_size = test_size if test_size is not None else self.params.test_size
        self.seed = seed if seed is not None else self.params.seed
        self.use_minibatch = use_minibatch

        self.best_n_clusters: Optional[int] = None
        self.best_e_rate: Optional[float] = None
        self.best_distance: float = np.inf

        # Detect region for distance cache
        self.region = getattr(self.instance_basic_data, "region", "pa-0")
        if not self.region or self.region == "pa-0":
            for part in str(getattr(self.instance_basic_data, "name", "")).split("_"):
                if "-" in part:
                    self.region = part
                    break

    def run(self) -> Tuple[KMeans, Dict[int, KMeans], List[int]]:
        """Executes offline calibration and trains the final 2-level spatial hierarchy."""
        coords = np.array([[pkg.point.lat, pkg.point.lng] for pkg in self.H])
        sizes = np.array([pkg.size for pkg in self.H])

        # Hyperparameter search grid
        c_grid = (
            list(self.n_clusters)
            if isinstance(self.n_clusters, (list, tuple, range))
            else [int(self.n_clusters)]
        )
        e_grid = (
            list(self.e_rate)
            if isinstance(self.e_rate, (list, tuple))
            else [float(self.e_rate)]
        )

        needs_search = len(c_grid) > 1 or len(e_grid) > 1

        if not needs_search:
            self.best_n_clusters = int(c_grid[0])
            self.best_e_rate = float(e_grid[0])
            return self._train_final_model(coords, self.best_n_clusters, self.best_e_rate)

        logger.info(
            f"[HCM-OFFLINE] Starting parameter calibration over "
            f"{len(c_grid)} cluster options x {len(e_grid)} e-rate options (seed={self.seed})..."
        )

        # Split historical data into training (H1) and validation (H2) sets
        indices = np.arange(len(self.H))
        idx_h1, idx_h2 = train_test_split(
            indices, test_size=self.test_size, random_state=self.seed
        )

        h1_coords = coords[idx_h1]
        h2_coords = coords[idx_h2]
        h2_sizes = sizes[idx_h2]

        origin_coord = np.array(
            [self.instance_basic_data.origin.lat, self.instance_basic_data.origin.lng]
        )

        # Precompute outlier filtering for each distinct e_rate
        clean_cache = {}
        for e in e_grid:
            clean_cache[e] = remove_outliers_isolation_forest(h1_coords, float(e), seed=self.seed)

        # Build task list
        tasks = []
        for c in c_grid:
            for e in e_grid:
                tasks.append(
                    (
                        int(c),
                        float(e),
                        clean_cache[e],
                        h2_coords,
                        h2_sizes,
                        self.n_unit_loads,
                        self.instance_basic_data.vehicle_capacity,
                        self.alpha_criteria,
                        origin_coord,
                        self.seed,
                        self.use_minibatch,
                        self.region,
                    )
                )

        num_workers = min(os.cpu_count() or 4, len(tasks))
        if num_workers > 1 and len(tasks) > 1:
            with Pool(processes=num_workers, initializer=_init_offline_worker) as pool:
                results = pool.map(evaluate_hcm_configuration, tasks)
        else:
            results = [evaluate_hcm_configuration(t) for t in tasks]

        # Find optimal configuration minimizing validation distance
        for dist, c, e in results:
            if dist < self.best_distance:
                self.best_distance = dist
                self.best_n_clusters = c
                self.best_e_rate = e

        logger.info(
            f"[HCM-OFFLINE] Calibration optimal: k1={self.best_n_clusters}, "
            f"e_rate={self.best_e_rate:.4f} (Validation Dist={self.best_distance:.2f} km)"
        )

        return self._train_final_model(coords, self.best_n_clusters, self.best_e_rate)

    def _train_final_model(
        self,
        coords: np.ndarray,
        k1_val: int,
        e_val: float,
    ) -> Tuple[KMeans, Dict[int, KMeans], List[int]]:
        """Fits the final 2-level hierarchy on full training coordinates."""
        # 1. Spatial Outlier Filtering
        points_clean = remove_outliers_isolation_forest(coords, e_val, seed=self.seed)

        # 2. Level-1 Macro Clustering
        k1 = min(k1_val, len(points_clean), self.n_unit_loads)
        if self.use_minibatch and len(points_clean) > 5000:
            clustering = MiniBatchKMeans(
                n_clusters=k1,
                random_state=self.seed,
                batch_size=2048,
                n_init=10,
                max_iter=100,
            )
        else:
            clustering = KMeans(
                n_clusters=k1,
                init="k-means++",
                random_state=self.seed,
                n_init="auto",
            )
        labels1 = clustering.fit_predict(points_clean)

        # 3. Unit Load Allocation (Max strategy)
        allocation, distribution_unit_loads = allocate_unit_loads(
            labels1, self.n_unit_loads
        )

        # 4. Level-2 Micro Clustering (Subclusters per Unit Load)
        subclusterings = {}
        for cluster_id, n_ul in allocation.items():
            pts_cluster = points_clean[labels1 == cluster_id]
            if len(pts_cluster) >= n_ul and n_ul >= 1:
                if self.use_minibatch and len(pts_cluster) > 5000:
                    sub_k = MiniBatchKMeans(
                        n_clusters=n_ul,
                        random_state=self.seed,
                        batch_size=1024,
                        n_init=10,
                        max_iter=100,
                    )
                else:
                    sub_k = KMeans(
                        n_clusters=n_ul,
                        init="k-means++",
                        random_state=self.seed,
                        n_init="auto",
                    )
                sub_k.fit(pts_cluster)
                subclusterings[cluster_id] = sub_k

        return clustering, subclusterings, distribution_unit_loads


# Backward compatibility
HCM_OFFLINE_V2 = HCM_OFFLINE
