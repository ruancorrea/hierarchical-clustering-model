import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
import numpy as np

from models import Point

logger = logging.getLogger(__name__)

# Global cache of OSRM availability per host (avoid repeating timeouts)
_OSRM_AVAILABILITY: Dict[str, bool] = {}


@dataclass
class OSRMConfig:
    host: str = "http://localhost:5000"
    timeout_s: int = 10
    cache_dir: Optional[Union[str, Path]] = None
    region: Optional[str] = None


def is_osrm_alive(host: str = "http://localhost:5000", sample_point: Optional[Point] = None) -> bool:
    """Check if OSRM backend is actually responding on the given host."""
    try:
        if sample_point is not None:
            resp = requests.get(f"{host}/nearest/v1/driving/{sample_point.lng},{sample_point.lat}", timeout=0.5)
        else:
            resp = requests.get(f"{host}/", timeout=0.5)
        # OSRM returns 200 or 400 (if root URL), both mean the server is running and alive
        return resp.status_code in [200, 400]
    except Exception:
        return False


def get_osrm_distance_matrix(
    points: List[Point], config: OSRMConfig
) -> np.ndarray:
    """Calculate distance matrix using OSRM table service."""
    coords_uri = ";".join([f"{point.lng},{point.lat}" for point in points])
    response = requests.get(
        f"{config.host}/table/v1/driving/{coords_uri}?annotations=distance",
        timeout=config.timeout_s,
    )
    response.raise_for_status()
    return np.array(response.json()["distances"], dtype=np.float64)


def get_osrm_route_distance(
    points: List[Point], config: OSRMConfig
) -> float:
    """Calculate route distance using OSRM route service."""
    coords_uri = ";".join(f"{point.lng},{point.lat}" for point in points)
    response = requests.get(
        f"{config.host}/route/v1/driving/{coords_uri}?annotations=distance&continue_straight=false",
        timeout=config.timeout_s,
    )
    response.raise_for_status()
    return float(min(r["distance"] for r in response.json()["routes"]))
