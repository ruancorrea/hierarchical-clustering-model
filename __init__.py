"""RFCT: Unified Framework for Online CVRP with Physical Unit Loads."""

from models import CVRPInstance, CVRPSolution, Delivery, Point
from core.engine import OnlineCVRPEngine
from core.base_policy import BaseRoutingPolicy
from policies import HCMPolicy, KMeansPolicy, SweepPolicy, GreedyTSPPolicy
from hcm import HCM_OFFLINE_V2, HCMParamsV2, HCM_ONLINE
from stats import StatisticalAnalyzer
from distances import preload_distance_cache

__all__ = [
    "CVRPInstance",
    "CVRPSolution",
    "Delivery",
    "Point",
    "OnlineCVRPEngine",
    "BaseRoutingPolicy",
    "HCMPolicy",
    "KMeansPolicy",
    "SweepPolicy",
    "GreedyTSPPolicy",
    "HCM_OFFLINE_V2",
    "HCMParamsV2",
    "StatisticalAnalyzer",
    "preload_distance_cache",
]
