"""Unit Load Allocation Module: Proportional Distribution with Max Remainder Strategy."""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def allocate_unit_loads(
    labels: np.ndarray,
    n_unit_loads: int,
) -> Tuple[Dict[int, int], List[int]]:
    """Distribute physical Unit Loads among Level-1 clusters proportionally to demand.

    Uses the largest remainder / greedy max strategy to guarantee exact sum(n_unit_loads)
    while ensuring every non-empty macro-cluster receives at least 1 Unit Load.

    Args:
        labels: (N,) array of Level-1 cluster labels.
        n_unit_loads: Total number of physical Unit Load sorting bays.

    Returns:
        Tuple of:
            - allocation: Dict mapping cluster_id -> number of allocated Unit Loads.
            - distribution_unit_loads: List of cluster IDs assigned to each sequential Unit Load.
    """
    unique_clusters, counts = np.unique(labels, return_counts=True)
    total_pts = np.sum(counts)
    allocation = defaultdict(int)
    total_alloc = 0

    # Initial proportional ceiling allocation (minimum 1 per cluster)
    for cluster_id, count in zip(unique_clusters, counts):
        pct = count / total_pts
        allocation[int(cluster_id)] = max(1, int(np.ceil(n_unit_loads * pct)))
        total_alloc += allocation[int(cluster_id)]

    # Greedy reduction from the cluster with highest allocation until total_alloc == n_unit_loads
    while total_alloc > n_unit_loads:
        eligible_clusters = [c for c in allocation if allocation[c] > 1]
        if not eligible_clusters:
            break
        max_cl = max(eligible_clusters, key=allocation.get)
        allocation[max_cl] -= 1
        total_alloc -= 1

    # Surplus distribution to the smallest cluster if needed
    while total_alloc < n_unit_loads:
        min_cl = min(allocation, key=allocation.get)
        allocation[min_cl] += 1
        total_alloc += 1

    # Map sequential Unit Load index (0 .. n_unit_loads - 1) to cluster ID
    distribution_unit_loads = []
    for cluster_id in sorted(allocation.keys()):
        n_ul = allocation[cluster_id]
        for _ in range(n_ul):
            distribution_unit_loads.append(cluster_id)

    return allocation, distribution_unit_loads


# Backward-compatibility alias
allocate_unit_loads_helper = lambda labels, n_unit_loads, strategy="greedy_max": allocate_unit_loads(labels, n_unit_loads)
