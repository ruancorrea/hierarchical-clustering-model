from typing import Iterable
import numpy as np

from models import Point

EARTH_RADIUS_METERS = 6371000


def calculate_distance_matrix_euclidean_m(points: Iterable[Point]) -> np.ndarray:
    """Calculate Great Circle / Spherical Euclidean distance matrix in meters."""
    points_list = list(points)
    if not points_list:
        return np.array([[]])

    points_rad = np.radians([(point.lat, point.lng) for point in points_list])

    delta_lambda = points_rad[:, [1]] - points_rad[:, 1]  # (N x M) lng
    phi1 = points_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = points_rad[:, 0]  # (1 x M) array of destination latitudes

    delta_sigma = np.arctan2(
        np.sqrt(
            (np.cos(phi2) * np.sin(delta_lambda)) ** 2
            + (
                np.cos(phi1) * np.sin(phi2)
                - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
            )
            ** 2
        ),
        (
            np.sin(phi1) * np.sin(phi2)
            + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
        ),
    )

    return EARTH_RADIUS_METERS * delta_sigma


def calculate_route_distance_euclidean_m(points: Iterable[Point]) -> float:
    """Calculate total route distance in meters using Great Circle / Spherical Euclidean metric."""
    points_list = list(points)
    if len(points_list) < 2:
        return 0.0

    distance_matrix = calculate_distance_matrix_euclidean_m(points_list)
    point_indices = np.arange(len(points_list))

    return float(distance_matrix[point_indices[:-1], point_indices[1:]].sum())
