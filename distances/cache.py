import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

from models import Point


logger = logging.getLogger(__name__)


def _clean_key(name_or_path: Union[str, Path]) -> str:
    """Normalize region or file name into a canonical key (e.g. 'pa-0' -> 'pa0')."""
    stem = Path(name_or_path).stem.lower()
    for suffix in ["_matrix", "_coords"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    if stem.startswith("cache_"):
        stem = stem[6:]
    return stem.replace("-", "").replace("_", "")


class DistanceCache:
    """In-memory or memory-mapped distance matrix cache (Singleton per dataset)."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.point_to_idx: Dict[Tuple[float, float], int] = {}
        self.matrix: Optional[np.ndarray] = None
        self.canonical_key: str = _clean_key(self.file_path)
        self._load()

    def _load(self):
        """Load distance cache from binary npy/npz or generate from parquet once."""
        stem = self.file_path.stem
        for suffix in ["_matrix", "_coords"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]

        parent = self.file_path.parent
        npy_matrix_path = parent / f"{stem}_matrix.npy"
        npy_coords_path = parent / f"{stem}_coords.npy"
        npz_path = parent / f"{stem}.npz"
        parquet_path = parent / f"{stem}.parquet"

        # 1. Try loading memory-mapped .npy (zero-copy mmap, fastest)
        if npy_matrix_path.is_file() and npy_coords_path.is_file():
            self.matrix = np.load(npy_matrix_path, mmap_mode="r")
            coords = np.load(npy_coords_path)
            self._build_index(coords)
            return

        # 2. Try loading compressed .npz
        if npz_path.is_file():
            data = np.load(npz_path)
            self.matrix = data["matrix"]
            coords = data["coords"]
            try:
                np.save(npy_matrix_path, self.matrix)
                np.save(npy_coords_path, coords)
            except Exception as e:
                logger.debug(f"Could not save npy sidecar: {e}")
            self._build_index(coords)
            return

        # 3. Fallback: load parquet and build sidecars once
        target_parquet = parquet_path if parquet_path.is_file() else self.file_path
        if target_parquet.is_file():
            logger.info(f"Generating binary distance cache for {stem} from parquet (one-time setup)...")
            import pandas as pd

            df = pd.read_parquet(
                target_parquet,
                columns=["p1_lat", "p1_lng", "p2_lat", "p2_lng", "distance_m"],
            )

            unique_pts_df = df[["p1_lat", "p1_lng"]].drop_duplicates().reset_index(drop=True)
            unique_pts_df["point_idx"] = unique_pts_df.index
            n_pts = len(unique_pts_df)

            p1_map = unique_pts_df.rename(columns={"point_idx": "p1_idx"})
            df = df.merge(p1_map, on=["p1_lat", "p1_lng"], how="left")

            p2_map = unique_pts_df.rename(
                columns={"p1_lat": "p2_lat", "p1_lng": "p2_lng", "point_idx": "p2_idx"}
            )
            df = df.merge(p2_map, on=["p2_lat", "p2_lng"], how="left")

            coords = unique_pts_df[["p1_lat", "p1_lng"]].to_numpy()
            matrix = np.zeros((n_pts, n_pts), dtype=np.float32)
            matrix[df["p1_idx"].to_numpy(), df["p2_idx"].to_numpy()] = df["distance_m"].to_numpy()

            try:
                np.save(npy_matrix_path, matrix)
                np.save(npy_coords_path, coords)
                np.savez_compressed(npz_path, matrix=matrix, coords=coords)
            except Exception as e:
                logger.warning(f"Could not save binary cache sidecars: {e}")

            self.matrix = matrix
            self._build_index(coords)
            return

        raise FileNotFoundError(f"No distance cache file found for {self.file_path}")

    def _build_index(self, coords: np.ndarray):
        """Build O(1) hash lookup supporting exact and rounded coordinate matching."""
        self.point_to_idx = {}
        for idx, (lat, lng) in enumerate(coords):
            f_lat, f_lng = float(lat), float(lng)
            self.point_to_idx[(f_lat, f_lng)] = idx
            self.point_to_idx[(round(f_lat, 6), round(f_lng, 6))] = idx
            self.point_to_idx[(round(f_lat, 5), round(f_lng, 5))] = idx

    def find_point_index(self, point: Point) -> Optional[int]:
        """Find the index of a point in the cache with O(1) lookup."""
        key = (float(point.lat), float(point.lng))
        if key in self.point_to_idx:
            return self.point_to_idx[key]

        r6 = (round(point.lat, 6), round(point.lng, 6))
        if r6 in self.point_to_idx:
            return self.point_to_idx[r6]

        r5 = (round(point.lat, 5), round(point.lng, 5))
        return self.point_to_idx.get(r5)

    def contains_all_points(self, points: Iterable[Point]) -> bool:
        """Check if all given points are present in the cache."""
        for p in points:
            if self.find_point_index(p) is None:
                return False
        return True

    def get_submatrix(self, points: List[Point]) -> Optional[np.ndarray]:
        """Return NxN submatrix for the given list of points, or None if any point is missing."""
        indices = []
        for p in points:
            idx = self.find_point_index(p)
            if idx is None:
                return None
            indices.append(idx)

        return np.asarray(self.matrix[np.ix_(indices, indices)], dtype=np.float64)

    def get_route_distance(self, points: List[Point]) -> Optional[float]:
        """Calculate route distance across sequential points, or None if missing."""
        indices = []
        for p in points:
            idx = self.find_point_index(p)
            if idx is None:
                return None
            indices.append(idx)

        idx_from = indices[:-1]
        idx_to = indices[1:]
        return float(self.matrix[idx_from, idx_to].sum())


# Global cache registry: canonical_key -> DistanceCache instance
_CACHE_REGISTRY: Dict[str, DistanceCache] = {}
_INITIALIZED_CANONICAL_KEYS: set = set()


def _register_cache(cache: DistanceCache, aliases: Optional[List[str]] = None):
    """Register cache under canonical key and all aliases to prevent re-initialization."""
    canon = cache.canonical_key
    _CACHE_REGISTRY[canon] = cache
    _INITIALIZED_CANONICAL_KEYS.add(canon)
    if aliases:
        for alias in aliases:
            if alias:
                _CACHE_REGISTRY[str(alias)] = cache
                _CACHE_REGISTRY[_clean_key(alias)] = cache


def clear_cache_registry():
    """Clear and close all registered distance caches and free memory."""
    global _CACHE_REGISTRY, _INITIALIZED_CANONICAL_KEYS
    for cache in list(_CACHE_REGISTRY.values()):
        if hasattr(cache, "matrix") and cache.matrix is not None:
            del cache.matrix
            cache.matrix = None
        if hasattr(cache, "point_to_idx"):
            cache.point_to_idx.clear()
    _CACHE_REGISTRY.clear()
    _INITIALIZED_CANONICAL_KEYS.clear()
    import gc
    gc.collect()


def get_cache_search_directories(cache_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Get list of potential directories where cache files may be located."""
    dirs = []
    if cache_dir:
        dirs.append(Path(cache_dir))

    env_cache = os.environ.get("CACHE_DIR")
    if env_cache:
        dirs.append(Path(env_cache))

    # Standard locations
    dirs.extend([
        Path("/cache"),
        Path(__file__).resolve().parent / "cache",
        Path(__file__).resolve().parent.parent / "cache",
        Path.cwd() / "cache",
        Path.cwd() / "app" / "cache",
        Path.cwd() / "app" / "distances" / "cache",
        Path.cwd() / "project" / "app" / "cache",
        Path.cwd() / "project" / "app" / "distances" / "cache",
    ])

    existing_dirs = []
    seen = set()
    for d in dirs:
        try:
            resolved = d.resolve()
            if resolved.is_dir() and resolved not in seen:
                seen.add(resolved)
                existing_dirs.append(resolved)
        except Exception:
            continue

    return existing_dirs


def find_cache_file_for_region(
    region: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """Find the best cache file for a region, prioritizing fast binary formats."""
    search_dirs = get_cache_search_directories(cache_dir)
    if not search_dirs or not region:
        return None

    clean = _clean_key(region)
    candidate_stems = [
        f"cache_{region}",
        f"cache_{region.lower()}",
        f"cache_{clean}",
        f"{region}",
        f"{region.lower()}",
        f"{clean}",
    ]

    for d in search_dirs:
        for stem in candidate_stems:
            for ext in ["_matrix.npy", ".npz", ".parquet"]:
                cand = d / f"{stem}{ext}"
                if cand.is_file():
                    return cand

        for f in d.glob("*"):
            if f.suffix in [".npy", ".npz", ".parquet"]:
                if clean in _clean_key(f):
                    return f

    return None


def preload_distance_cache(
    region: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
) -> Optional[DistanceCache]:
    """Ensure distance cache is loaded exactly once into the global registry."""
    if region:
        clean = _clean_key(region)
        if clean in _CACHE_REGISTRY:
            return _CACHE_REGISTRY[clean]
        if region in _CACHE_REGISTRY:
            return _CACHE_REGISTRY[region]

    # Check if ANY cache is already registered
    if not region and _CACHE_REGISTRY:
        return next(iter(_CACHE_REGISTRY.values()))

    # Find file specifically matching requested region
    cache_file = find_cache_file_for_region(region=region, cache_dir=cache_dir)
    if not cache_file and not region:
        # Only search generic dirs if NO specific region was requested
        search_dirs = get_cache_search_directories(cache_dir)
        for d in search_dirs:
            for f in d.glob("*"):
                if f.suffix in [".npy", ".npz", ".parquet"]:
                    cache_file = f
                    break
            if cache_file:
                break

    if cache_file:
        canon = _clean_key(cache_file)
        if canon in _CACHE_REGISTRY:
            return _CACHE_REGISTRY[canon]
        cache = DistanceCache(cache_file)
        _register_cache(cache, aliases=[region, str(cache_file), str(cache_file.resolve())])
        return cache

    return None


def get_distance_between_points(
    p1: Point, p2: Point, region: Optional[str] = None
) -> float:
    """Get real road network distance (in meters) between two points from cache with Euclidean fallback."""
    cache = preload_distance_cache(region=region)
    if cache is not None:
        i1 = cache.find_point_index(p1)
        i2 = cache.find_point_index(p2)
        if i1 is not None and i2 is not None and cache.matrix is not None:
            return float(cache.matrix[i1, i2])

    d_lat = (p1.lat - p2.lat) * 111_000.0
    d_lng = (p1.lng - p2.lng) * 111_000.0
    return float(np.sqrt(d_lat ** 2 + d_lng ** 2))



def get_distance_cache(
    points: Iterable[Point],
    region: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[DistanceCache]:
    """Retrieve or lazily initialize the distance cache once."""
    points_list = list(points)

    # 1. Fast check: return already initialized cache if it matches points
    if region:
        clean = _clean_key(region)
        if clean in _CACHE_REGISTRY:
            cache = _CACHE_REGISTRY[clean]
            if cache.contains_all_points(points_list):
                return cache

    for cache in _CACHE_REGISTRY.values():
        if cache.contains_all_points(points_list):
            return cache

    # 2. Lazily load via preload_distance_cache (guaranteed once per dataset)
    cache = preload_distance_cache(region=region, cache_dir=cache_dir)
    if cache and cache.contains_all_points(points_list):
        return cache

    return None
