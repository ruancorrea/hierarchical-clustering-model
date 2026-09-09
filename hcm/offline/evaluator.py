"""Validation Performance Evaluator for HCM Hyperparameter Selection (H2)."""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from models import Point
from distances import calculate_route_distance_m
from hcm.offline.allocation import allocate_unit_loads


def fast_route_distance_estimator(
    points: np.ndarray, origin: np.ndarray, region: Optional[str] = None
) -> float:
    """Fast TSP route distance estimator using nearest-neighbor greedy ordering."""
    n = len(points)
    if n == 0:
        return 0.0
    if n == 1:
        route_pts = [
            Point(float(origin[0]), float(origin[1])),
            Point(float(points[0][0]), float(points[0][1])),
            Point(float(origin[0]), float(origin[1])),
        ]
        return float(calculate_route_distance_m(route_pts, region=region)) / 1000.0

    unvisited = list(range(n))
    curr_pos = origin
    ordered_indices = []

    while unvisited:
        dists = np.sum((points[unvisited] - curr_pos) ** 2, axis=1)
        nearest_idx = np.argmin(dists)
        chosen = unvisited.pop(nearest_idx)
        ordered_indices.append(chosen)
        curr_pos = points[chosen]

    route_pts = (
        [Point(float(origin[0]), float(origin[1]))]
        + [Point(float(points[i][0]), float(points[i][1])) for i in ordered_indices]
        + [Point(float(origin[0]), float(origin[1]))]
    )

    return float(calculate_route_distance_m(route_pts, region=region)) / 1000.0


def evaluate_hcm_configuration(args_tuple) -> Tuple[float, int, float]:
    """Worker function to train (n_clusters, e_rate) and evaluate total validation distance."""
    (
        c_clusters,
        e_rate,
        points_clean,
        h2_coords,
        h2_sizes,
        n_unit_loads,
        vehicle_capacity,
        alpha_criteria,
        origin_coord,
        seed,
        use_minibatch,
        region,
    ) = args_tuple

    # 1. Level-1 Clustering
    k1 = min(c_clusters, len(points_clean), n_unit_loads)
    if use_minibatch and len(points_clean) > 5000:
        km1 = MiniBatchKMeans(
            n_clusters=k1,
            random_state=seed,
            batch_size=2048,
            n_init=10,
            max_iter=100,
        )
    else:
        km1 = KMeans(n_clusters=k1, init="k-means++", random_state=seed, n_init="auto")
    labels1 = km1.fit_predict(points_clean)

    # 2. Unit Load Allocation (Max strategy)
    allocation, distribution_unit_loads = allocate_unit_loads(labels1, n_unit_loads)

    # 3. Level-2 Clustering (Subclusters)
    subclusterings = {}
    for cluster_id, n_ul in allocation.items():
        pts_c = points_clean[labels1 == cluster_id]
        if len(pts_c) >= n_ul and n_ul >= 1:
            if use_minibatch and len(pts_c) > 5000:
                sub_km = MiniBatchKMeans(
                    n_clusters=n_ul,
                    random_state=seed,
                    batch_size=1024,
                    n_init=10,
                    max_iter=100,
                )
            else:
                sub_km = KMeans(n_clusters=n_ul, init="k-means++", random_state=seed, n_init="auto")
            sub_km.fit(pts_c)
            subclusterings[cluster_id] = sub_km

    # 4. Fast Online Simulation on Validation Set (H2)
    # Direct 2-level cluster assignment (no proximity beta)
    val_labels1 = km1.predict(h2_coords)
    assigned_uls = np.zeros(len(h2_coords), dtype=int)

    for cluster_id in np.unique(val_labels1):
        mask = val_labels1 == cluster_id
        pts_in_c = h2_coords[mask]
        if len(pts_in_c) == 0:
            continue
        base_ul = distribution_unit_loads.index(cluster_id)
        if cluster_id in subclusterings:
            sub_preds = subclusterings[cluster_id].predict(pts_in_c)
            assigned_uls[mask] = np.clip(base_ul + sub_preds, 0, n_unit_loads - 1)
        else:
            assigned_uls[mask] = np.clip(base_ul, 0, n_unit_loads - 1)

    # Route simulation and capacity management
    total_dist = 0.0
    for ul_idx in range(n_unit_loads):
        ul_mask = assigned_uls == ul_idx
        if not np.any(ul_mask):
            continue
        ul_pts = h2_coords[ul_mask]
        ul_szs = h2_sizes[ul_mask]

        curr_load = 0
        curr_route_pts = []
        for pt, sz in zip(ul_pts, ul_szs):
            if curr_load + sz > vehicle_capacity * alpha_criteria and curr_load > 0:
                total_dist += fast_route_distance_estimator(np.array(curr_route_pts), origin_coord, region=region)
                curr_route_pts = [pt]
                curr_load = sz
            else:
                curr_route_pts.append(pt)
                curr_load += sz
        if curr_route_pts:
            total_dist += fast_route_distance_estimator(np.array(curr_route_pts), origin_coord, region=region)

    return total_dist, c_clusters, e_rate


# Alias for backward compatibility
evaluate_hcm_configuration_fast = lambda args: [evaluate_hcm_configuration(args)]
