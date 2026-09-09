from typing import List, Optional, Dict
import random 
import math
import numpy as np
from dataclasses import dataclass

from models import (
    JSONDataclassMixin,
    CVRPInstance,
    CVRPSolution,
    Delivery,
    CVRPSolutionVehicle,
    Point
)

import os
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, Manager
import time
import numpy as np
import pandas as pd

from loggibud.v1.eval.task1 import evaluate_solution

from shared.ortools import (
    solve as ortools_solve,
    ORToolsParams
)

@dataclass
class Params(JSONDataclassMixin):
    ortools_tsp_params: Optional[ORToolsParams] = ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            )
    num_loadings_units: Optional[int] = 28
    seed: int = 0

    @classmethod
    def get_baseline(cls):
        return cls(
            seed = 0,
            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )
    
    @classmethod
    def create_dict(cls, p: Dict):
        return cls(
            num_loadings_units=p['num_loadings_units'],
            seed = 0,
            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )

@dataclass
class LoadingUnitModel:
    capacity: int
    deliveries: List[Delivery]
    distance: float

    @classmethod
    def get_baseline(cls):
        return cls(
            capacity= 0,
            distance=0.0,
            deliveries= []
        )

def __create_instanceCVRP(instance, deliveries: List[Delivery]) -> CVRPInstance:
    """ Creating an instanceCVRP with deliveries. """

    return CVRPInstance(
                name = instance.name,
                region= "",
                origin= instance.origin,
                vehicle_capacity = 2*instance.vehicle_capacity,
                deliveries= deliveries
            )

def solve_tsp(dispatch: CVRPInstance, params) -> CVRPSolutionVehicle:
    """ Generate route from a tsp solution. """
    solution=None
    while True:
        solution=ortools_solve(dispatch, params.ortools_tsp_params)# TSP
        if isinstance(solution, CVRPSolution):
            break
        else:
            random.shuffle(dispatch.deliveries)
    return CVRPSolutionVehicle(dispatch.origin, solution.deliveries)

def custo(instance, LU, lu_list, params):
    
    distance_with_delivery= 0
    if len(LU.deliveries) > 0:
        instance_lu =__create_instanceCVRP(instance, lu_list)
        solution_lu = CVRPSolution(name=instance.name,
                                vehicles=[ solve_tsp(instance_lu, params) ]
                                    )
        distance_with_delivery=evaluate_solution(instance_lu, solution_lu)
        

    return distance_with_delivery-LU.distance, distance_with_delivery

def allocation(instance, params):
    dispatches = []
    LUs = [LoadingUnitModel.get_baseline() for i in range(params.num_loadings_units)]
    ucs_utilizadas = {i: 0 for i in range(params.num_loadings_units)}
    for i, delivery in enumerate(instance.deliveries):
        min_distance = np.inf
        min_lu_index = -1
        for index, lu in enumerate(LUs):
            lu_list = lu.deliveries.copy()
            lu_list.append(delivery)
            diff_tsp, distance_with_delivery = custo(instance, lu, lu_list, params)
            if diff_tsp < min_distance:
                min_distance = diff_tsp                
                min_lu_index = index
                distance=distance_with_delivery
        ucs_utilizadas[min_lu_index] += 1
        LUs[min_lu_index].capacity += delivery.size
        LUs[min_lu_index].deliveries.append(delivery)
        
        if LUs[min_lu_index].capacity >= instance.vehicle_capacity *0.95:
            dispatches.append(LUs[min_lu_index].deliveries)
            LUs[min_lu_index]=LoadingUnitModel.get_baseline()
        LUs[min_lu_index].distance=distance if len(LUs[min_lu_index].deliveries) > 1 else min_distance
        
        print(instance.name, i, ucs_utilizadas)
    for loading_unit in LUs:
        if len(loading_unit.deliveries)>0:
            dispatches.append(loading_unit.deliveries)

    return CVRPSolution(
        name=instance.name,
        vehicles=[ solve_tsp(
                    __create_instanceCVRP(instance, dispatch), 
                    params) 
                    for dispatch in dispatches ],
    )

if __name__ == "__main__":
    #regions = ['pa-0', 'pa-1','df-0', 'df-1', 'df-2']
    #regions = ['rj-0', 'rj-1', 'rj-2']
    #regions = ['rj-3', 'rj-4']
    regions = ['rj-5']
    for dir in regions:
        params = {"num_loadings_units": 28}
        params = Params.create_dict(params) if params else Params.get_baseline()
        print(dir, params)

        paths_instances_test = f'data/cvrp-instances-1.0/dev/{dir}'
        paths_instances_test = f'batches/instances/{dir}'
        test_path = Path(paths_instances_test)
        test_path_dir = test_path if test_path.is_dir() else test_path.parent
        test_files = (
            [test_path] if test_path.is_file() else list(test_path.iterdir())
        )

        eval_files = [CVRPInstance.from_file(f) for f in test_files[:240]]


        manager = Manager()
        results = manager.list()


        def solve(file):
            instance = CVRPInstance.from_file(file)
            instance.vehicle_capacity=180
            start = time.perf_counter()
            solution = allocation(instance,params)
            end = time.perf_counter()
            distance = evaluate_solution(instance, solution)
            print(instance.name, distance, (end - start)/60)
            res = [instance.name, distance, (end - start)/60, len(solution.vehicles)]
            results.append(res)
            path_csv = f"./results/greedy_tsp/{dir}_{params.num_loadings_units}.csv"
            df = pd.DataFrame([{'path': item[0], 'distance': item[1], 'time_test_model': item[2], 'vehicles': item[3]} for item in results])
            df.to_csv(Path(path_csv))

        with Pool(os.cpu_count()) as pool:
            list(tqdm(pool.imap(solve, test_files), total=len(test_files)))
