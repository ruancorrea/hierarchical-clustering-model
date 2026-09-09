from abc import ABC, abstractmethod
from typing import Dict, List, Any
from models import CVRPInstance, Delivery


class BaseRoutingPolicy(ABC):
    """Abstract Base Class for CVRP Online Routing Policies / Solution Methods."""

    def __init__(self, name: str = "BasePolicy"):
        self.name = name
        self.n_unit_loads: int = 28

    @abstractmethod
    def fit(self, train_instances: List[CVRPInstance], n_unit_loads: int, **kwargs) -> "BaseRoutingPolicy":
        """Fase Offline: Treina parâmetros a partir do histórico de instâncias de entrega."""
        pass

    @abstractmethod
    def select_unit_load(
        self,
        delivery: Delivery,
        unit_loads_state: Dict[int, List[Delivery]],
        capacities: Dict[int, int],
        vehicle_capacity: int,
    ) -> int:
        """Fase Online (Passo 2): Retorna o índice da Unit Load (0 .. n_unit_loads - 1) para o novo pacote."""
        pass
