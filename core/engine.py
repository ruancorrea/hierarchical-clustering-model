import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

from models import (
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
    Delivery,
)
from shared.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)
from eval import evaluate_solution
from utils import create_CVRPInstance
from core.base_policy import BaseRoutingPolicy

logger = logging.getLogger(__name__)


class OnlineCVRPEngine:
    """General Online CVRP Lifecycle Engine managing N physical Unit Loads."""

    def __init__(
        self,
        policy: BaseRoutingPolicy,
        n_unit_loads: int = 28,
        alpha_criteria: float = 0.94,
        time_limit_ms_tsp: int = 1_000,
        show_progress: bool = False,
    ):
        self.policy = policy
        self.n_unit_loads = n_unit_loads
        self.alpha_criteria = alpha_criteria
        self.time_limit_ms_tsp = time_limit_ms_tsp
        self.show_progress = show_progress
        self.ortools_tsp_params = ORToolsParams(
            max_vehicles=1,
            time_limit_ms=time_limit_ms_tsp,
        )

    def _solve_tsp(self, deliveries: List[Delivery], instance: CVRPInstance) -> CVRPSolutionVehicle:
        """Solve Travelling Salesperson Problem (TSP) for a closed route."""
        if not deliveries:
            return CVRPSolutionVehicle(instance.origin, [])
        if len(deliveries) == 1:
            return CVRPSolutionVehicle(instance.origin, deliveries)

        subinstance = create_CVRPInstance(instance, deliveries, factor=3)
        solution = ortools_solve(subinstance, self.ortools_tsp_params)
        if isinstance(solution, CVRPSolution) and solution.vehicles:
            return CVRPSolutionVehicle(instance.origin, solution.deliveries)
        return CVRPSolutionVehicle(instance.origin, deliveries)

    def run(self, instance: CVRPInstance) -> Tuple[CVRPSolution, float]:
        """Execute the 4-step Online CVRP lifecycle on the given instance.
        
        FASE ONLINE:
         1. Para cada novo pacote
         2. Método de Solução retorna qual a sua unidade de carregamento
         3. Alocação do pedido na unidade de carregamento
         4. Análise de critérios de fechamento (alpha_criteria)
        """
        # Inicializa as N posições físicas de Unit Loads
        unit_loads_packages: Dict[int, List[Delivery]] = {
            i: [] for i in range(self.n_unit_loads)
        }
        unit_loads_capacity: Dict[int, int] = {
            i: 0 for i in range(self.n_unit_loads)
        }
        routes: List[List[Delivery]] = []

        deliveries_iter = (
            tqdm(instance.deliveries, desc=f"Online routing ({self.policy.name})")
            if self.show_progress
            else instance.deliveries
        )

        # FASE ONLINE:
        for delivery in deliveries_iter:
            # 1. Novo pacote recebido
            # 2. Método de Solução retorna a Unit Load
            unit_load = self.policy.select_unit_load(
                delivery=delivery,
                unit_loads_state=unit_loads_packages,
                capacities=unit_loads_capacity,
                vehicle_capacity=instance.vehicle_capacity,
            )

            # 3. Alocação do pedido na Unit Load respeitando a capacidade máxima física
            if unit_loads_capacity[unit_load] + delivery.size > instance.vehicle_capacity:
                if unit_loads_packages[unit_load]:
                    routes.append(unit_loads_packages[unit_load])
                unit_loads_packages[unit_load] = [delivery]
                unit_loads_capacity[unit_load] = delivery.size
            else:
                unit_loads_packages[unit_load].append(delivery)
                unit_loads_capacity[unit_load] += delivery.size

            # 4. Análise de critérios de fechamento (alpha_criteria, ex: >= 94%)
            if unit_loads_capacity[unit_load] >= instance.vehicle_capacity * self.alpha_criteria:
                routes.append(unit_loads_packages[unit_load])
                unit_loads_packages[unit_load] = []
                unit_loads_capacity[unit_load] = 0

        # Fechamento de todas as Unit Loads residuais que possuem pacotes
        for unit_load, packages in unit_loads_packages.items():
            if unit_loads_capacity[unit_load] > 0 and len(packages) > 0:
                routes.append(packages)

        # Otimização TSP de cada rota
        vehicles: List[CVRPSolutionVehicle] = []
        routes_iter = (
            tqdm(routes, desc=f"TSP Optimization ({self.policy.name})")
            if self.show_progress
            else routes
        )
        for route_deliveries in routes_iter:
            vehicles.append(self._solve_tsp(route_deliveries, instance))

        solution = CVRPSolution(name=instance.name, vehicles=vehicles)
        distance = evaluate_solution(instance, solution)

        return solution, distance
