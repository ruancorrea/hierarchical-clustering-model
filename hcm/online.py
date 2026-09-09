"""Online Real-Time Routing Engine for Hierarchical Cluster Model (HCM)."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from shared.ortools import solve as ortools_solve, ORToolsParams
from models import CVRPInstance, CVRPSolution, CVRPSolutionVehicle, Delivery
from eval import evaluate_solution
from utils import create_CVRPInstance

logger = logging.getLogger(__name__)


class HCM_ONLINE:
    """Online real-time vehicle route generation using trained 2-level spatial clusters."""

    def __init__(
        self,
        n_unit_loads: int,
        data: CVRPInstance,
        clustering: KMeans,
        subclusterings: Dict[int, KMeans],
        alpha_criteria: float = 0.94,
        distribution_unit_loads: Optional[List[int]] = None,
        time_limit_ms_tsp: int = 1_000,
        **kwargs,
    ):
        self.n_unit_loads = n_unit_loads
        self.data = data
        self.clustering = clustering
        self.subclusterings = subclusterings
        self.alpha_criteria = alpha_criteria
        self.distribution_unit_loads = distribution_unit_loads or list(range(n_unit_loads))
        self.ortools_tsp_params = ORToolsParams(
            max_vehicles=1,
            time_limit_ms=time_limit_ms_tsp,
        )
        self.routes = []
        self.solution = None
        self.distance = 0.0

    def define_unit_load_of_package(self, current_point: np.ndarray) -> int:
        """Assign package to Unit Load via 2-level spatial hierarchy."""
        cluster = int(self.clustering.predict([current_point])[0])
        sub_index = 0
        if cluster in self.subclusterings:
            sub_index = int(self.subclusterings[cluster].predict([current_point])[0])

        base_ul = self.distribution_unit_loads.index(cluster)
        selected_ul = base_ul + sub_index
        return min(max(selected_ul, 0), self.n_unit_loads - 1)

    def generate_route(self, instance: CVRPInstance) -> CVRPSolutionVehicle:
        """Generate optimized TSP vehicle route."""
        solution = ortools_solve(instance, self.ortools_tsp_params)
        if isinstance(solution, CVRPSolution):
            return CVRPSolutionVehicle(instance.origin, solution.deliveries)
        return CVRPSolutionVehicle(instance.origin, instance.deliveries)

    def run(self) -> Tuple[CVRPSolution, float]:
        """Run online package allocation and intra-cluster route generation."""
        unit_loads_packages: Dict[int, List[Delivery]] = {ul: [] for ul in range(self.n_unit_loads)}
        unit_loads_capacity: Dict[int, int] = {ul: 0 for ul in range(self.n_unit_loads)}
        routes = []

        for package in self.data.deliveries:
            pt = np.array([package.point.lat, package.point.lng])
            ul = self.define_unit_load_of_package(pt)
            unit_loads_packages[ul].append(package)
            unit_loads_capacity[ul] += package.size

            if unit_loads_capacity[ul] >= self.data.vehicle_capacity * self.alpha_criteria:
                routes.append(unit_loads_packages[ul])
                unit_loads_packages[ul] = []
                unit_loads_capacity[ul] = 0

        for ul, packages in unit_loads_packages.items():
            if packages:
                routes.append(packages)

        vehicles = []
        for route in routes:
            vehicles.append(self.generate_route(create_CVRPInstance(self.data, route)))

        self.solution = CVRPSolution(name=self.data.name, vehicles=vehicles)
        self.distance = evaluate_solution(self.data, self.solution)
        return self.solution, self.distance


# Aliases
HCMOnlineSolver = HCM_ONLINE
