import logging
from typing import Iterable, Optional, Union
import numpy as np
import requests

from models import Point
from distances.osrm import OSRMConfig, get_osrm_distance_matrix, get_osrm_route_distance
from distances.cache import get_distance_cache
from distances.euclidean import (
    calculate_distance_matrix_euclidean_m,
    calculate_route_distance_euclidean_m,
)


logger = logging.getLogger(__name__)


class DistanceController:
    """Controller managing distance calculations via OSRM, Parquet Cache, or Euclidean metric."""

    @staticmethod
    def calculate_distance_matrix_m(
        points: Iterable[Point],
        config: Optional[OSRMConfig] = None,
        region: Optional[str] = None,
    ) -> Union[int, np.ndarray]:
        """Calculate distance matrix in meters with automatic fallback."""
        points_list = list(points)
        if len(points_list) < 2:
            return 0

        config = config or OSRMConfig()
        target_region = region or config.region

        # 1. Try OSRM server
        try:
            return get_osrm_distance_matrix(points_list, config)
        except (requests.RequestException, Exception) as exc:
            logger.debug(f"OSRM server unavailable ({exc}). Falling back to cache / euclidean.")

        # 2. Check parquet cache
        cache = get_distance_cache(
            points_list, region=target_region, cache_dir=config.cache_dir
        )
        if cache is not None:
            submatrix = cache.get_submatrix(points_list)
            if submatrix is not None:
                return submatrix

        # 3. Fallback to Euclidean / Great Circle metric
        return calculate_distance_matrix_euclidean_m(points_list)

    @staticmethod
    def calculate_route_distance_m(
        points: Iterable[Point],
        config: Optional[OSRMConfig] = None,
        region: Optional[str] = None,
    ) -> Union[int, float]:
        """Calculate route total distance in meters with automatic fallback."""
        points_list = list(points)
        if len(points_list) < 2:
            return 0

        config = config or OSRMConfig()
        target_region = region or config.region

        # 1. Try OSRM server
        try:
            return get_osrm_route_distance(points_list, config)
        except (requests.RequestException, Exception) as exc:
            logger.debug(f"OSRM server unavailable ({exc}). Falling back to cache / euclidean.")

        # 2. Check parquet cache
        cache = get_distance_cache(
            points_list, region=target_region, cache_dir=config.cache_dir
        )
        if cache is not None:
            route_dist = cache.get_route_distance(points_list)
            if route_dist is not None:
                return float(route_dist)

        # 3. Fallback to Euclidean / Great Circle metric
        return float(calculate_route_distance_euclidean_m(points_list))


# Helper functions exposing the Controller API directly
def calculate_distance_matrix_m(
    points: Iterable[Point],
    config: Optional[OSRMConfig] = None,
    region: Optional[str] = None,
) -> Union[int, np.ndarray]:
    return DistanceController.calculate_distance_matrix_m(points, config=config, region=region)


def calculate_route_distance_m(
    points: Iterable[Point],
    config: Optional[OSRMConfig] = None,
    region: Optional[str] = None,
) -> Union[int, float]:
    return DistanceController.calculate_route_distance_m(points, config=config, region=region)


def get_active_distance_metric(
    region: Optional[str] = None,
    config: Optional[OSRMConfig] = None,
) -> str:
    """Determine the active distance metric: 'osrm', 'cache', or 'euclidean'."""
    cfg = config or OSRMConfig()
    from distances.osrm import is_osrm_alive
    if is_osrm_alive(cfg.host):
        return "osrm"

    target_region = region or cfg.region
    from distances.cache import preload_distance_cache
    cache = preload_distance_cache(region=target_region, cache_dir=cfg.cache_dir)
    if cache is not None and cache.matrix is not None:
        return "cache"

    return "euclidean"
