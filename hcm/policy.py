"""Hierarchical Cluster Model (HCM) Routing Policy."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from models import CVRPInstance, Delivery
from core.base_policy import BaseRoutingPolicy
from hcm.offline.outliers import remove_outliers_isolation_forest
from hcm.offline.allocation import allocate_unit_loads
from hcm.offline.core import HCM_OFFLINE, HCMParams

logger = logging.getLogger(__name__)


class HCMPolicy(BaseRoutingPolicy):
    """Hierarchical Cluster Model (HCM) Solution Method.

    Decomposes the dynamic online CVRP into:
    1. Offline Spatial Macro-Clustering (Level 1: k1 clusters)
    2. Proportional Unit Load Allocation (Max Remainder Strategy)
    3. Offline Spatial Micro-Clustering (Level 2: subclusters per Unit Load)
    4. Fast Online 2-Level Hierarchical Assignment
    """

    def __init__(
        self,
        n_clusters: Union[int, List[int], range] = 8,
        e_rate: Union[float, List[float]] = 0.0,
        n_unit_loads: int = 28,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(name="HCM")
        self.n_clusters = n_clusters
        self.e_rate = e_rate
        self.n_unit_loads = n_unit_loads
        self.seed = seed

        self.clustering: Optional[KMeans] = None
        self.subclusterings: Dict[int, KMeans] = {}
        self.distribution_unit_loads: List[int] = []

    def fit(self, train_instances: List[CVRPInstance], n_unit_loads: Optional[int] = None, **kwargs) -> "HCMPolicy":
        """Offline Phase: Fits the 2-level spatial hierarchy and distributes Unit Loads."""
        if n_unit_loads is not None:
            self.n_unit_loads = n_unit_loads

        # If hyperparameter search grid is provided (list/range), run offline calibration
        is_search = (
            isinstance(self.n_clusters, (list, tuple, range))
            or isinstance(self.e_rate, (list, tuple))
        )
        if is_search:
            params = HCMParams(
                n_clusters=self.n_clusters,
                e_rate=self.e_rate,
                n_unit_loads=self.n_unit_loads,
                seed=self.seed,
            )
            offline_runner = HCM_OFFLINE(data=train_instances, params=params)
            self.clustering, self.subclusterings, self.distribution_unit_loads = offline_runner.run()
            self.n_clusters = getattr(offline_runner, "best_n_clusters", self.clustering.n_clusters)
            self.e_rate = getattr(offline_runner, "best_e_rate", self.e_rate)
            return self

        # Direct fit on full historical data
        all_deliveries = [d for inst in train_instances for d in inst.deliveries]
        points = np.array([[d.point.lat, d.point.lng] for d in all_deliveries])

        # 1. Spatial Outlier Filtering (Isolation Forest)
        points_clean = remove_outliers_isolation_forest(points, float(self.e_rate), seed=self.seed)

        # 2. Level-1 Macro Clustering
        k1 = min(int(self.n_clusters), len(points_clean), self.n_unit_loads)
        if len(points_clean) > 5000:
            self.clustering = MiniBatchKMeans(
                n_clusters=k1,
                random_state=self.seed,
                batch_size=2048,
                n_init=10,
                max_iter=100,
            )
        else:
            self.clustering = KMeans(
                n_clusters=k1,
                init="k-means++",
                random_state=self.seed,
                n_init="auto",
            )
        labels1 = self.clustering.fit_predict(points_clean)

        # 3. Proportional Unit Load Allocation (Max strategy)
        allocation, self.distribution_unit_loads = allocate_unit_loads(
            labels1, self.n_unit_loads
        )

        # 4. Level-2 Micro Clustering (Subclusters per Unit Load)
        self.subclusterings = {}
        for cluster_id, n_ul in allocation.items():
            pts_cluster = points_clean[labels1 == cluster_id]
            if len(pts_cluster) >= n_ul and n_ul >= 1:
                if len(pts_cluster) > 5000:
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
                self.subclusterings[cluster_id] = sub_k

        logger.info(
            f"HCMPolicy fitted (k1={k1}, e_rate={self.e_rate}, "
            f"n_unit_loads={self.n_unit_loads}, subclusters={len(self.subclusterings)})."
        )
        return self

    def train(self, train_instances: List[CVRPInstance], n_unit_loads: Optional[int] = None, **kwargs) -> "HCMPolicy":
        """Alias for fit()."""
        return self.fit(train_instances, n_unit_loads=n_unit_loads, **kwargs)

    def select_unit_load(
        self,
        delivery: Delivery,
        unit_loads_state: Dict[int, List[Delivery]],
        capacities: Dict[int, int],
        vehicle_capacity: int,
    ) -> int:
        """Online Phase: Direct 2-level hierarchical cluster assignment."""
        curr_pt = np.array([[delivery.point.lat, delivery.point.lng]])

        # 1. Level-1 Macro Cluster Assignment
        cluster = int(self.clustering.predict(curr_pt)[0])
        sub_idx = 0
        if cluster in self.subclusterings:
            sub_idx = int(self.subclusterings[cluster].predict(curr_pt)[0])

        # 2. Map sub-cluster to physical Unit Load index (0 .. n_unit_loads - 1)
        base_ul_idx = self.distribution_unit_loads.index(cluster)
        selected_ul = base_ul_idx + sub_idx

        return min(max(selected_ul, 0), self.n_unit_loads - 1)
