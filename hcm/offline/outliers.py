"""Spatial Outlier Filtering Module using Isolation Forest."""

import numpy as np
from sklearn.ensemble import IsolationForest


def remove_outliers_isolation_forest(points: np.ndarray, rate: float, seed: int = 42) -> np.ndarray:
    """Remove spatial outliers from historical delivery points using Isolation Forest.

    Args:
        points: (N, 2) array of latitude and longitude coordinates.
        rate: Contamination rate e in [0.0, 0.5]. If rate <= 0.0, points are unmodified.
        seed: Random seed for deterministic reproducibility.

    Returns:
        (M, 2) filtered array of spatial coordinates.
    """
    if rate <= 0.0 or len(points) < 10:
        return points

    iso = IsolationForest(contamination=rate, random_state=seed, n_jobs=1)
    outliers = iso.fit_predict(points)
    return points[outliers == 1]
