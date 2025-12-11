import os
from typing import List

import yaml
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.routing.query_router import QueryRouter
from autoslo.strategies.slo_strategy import SLOStrategy
from autoslo.strategies.slo_strategy_performance import SLOStrategyPerformance
from autoslo.workload_definition.composite import Composite
from autoslo.workload_execution.trace import Trace

router = APIRouter()


@router.get("/composite", response_model=List[str])
def list_composite_workloads():
    """
    List the names of available composite workloads.

    Returns:
        A list of composite workload names.
    """
    base = pu.get_data_path()

    p = os.path.join(base, "composite_workloads")
    if not os.path.exists(p):
        return []
    return sorted(
        [f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f))]
    )


@router.get("/composite/{name}", response_model=dict)
def get_composite_workload(name: str):
    """
    For the named workload, return its definition as a dictionary, based on
    its definition.yml file.

    Parameters:
        name: The name of the composite workload.

    Returns:
        The workload definition as a dictionary.

    Raises:
        HTTPException: If the workload or its definition file is not found.
    """
    base = pu.get_data_path()
    workload_dir = os.path.join(base, "composite_workloads", name)
    if not os.path.exists(workload_dir):
        raise HTTPException(status_code=404, detail="Workload not found")

    def_file = os.path.join(workload_dir, "definition.yml")
    if not os.path.exists(def_file):
        raise HTTPException(status_code=404, detail="Definition file not found")

    with open(def_file, "r") as f:
        definition = yaml.safe_load(f)

    return definition


@router.post("/composite/create", response_model=str)
def create_composite_workload_def_post(workload_definition: dict = Body(...)):
    """
    Create a new composite workload definition (POST JSON).
    """

    # Check if a name is provided and doesn't exist already
    name = workload_definition.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Workload name is required")
    base = pu.get_data_path()
    workload_dir = os.path.join(base, "composite_workloads", name)
    if os.path.exists(workload_dir):
        raise HTTPException(status_code=400, detail="Workload already exists")

    # Normalize monday_index
    mi = workload_definition.get("monday_index", 0)
    if not isinstance(mi, int):
        try:
            mi = int(mi)
        except Exception:
            mi = 0
    workload_definition["monday_index"] = mi % 7

    # Validate and normalize "days" to list of dicts with "chunks"
    days = workload_definition.get("days")
    if not isinstance(days, list):
        raise HTTPException(
            status_code=400,
            detail='"days" array is required in workload definition',
        )

    for day in days:
        if isinstance(day, dict):
            chunks = day.get("chunks")
        else:
            raise HTTPException(
                status_code=400,
                detail='Each "day" must be an object with "chunks" array or a list of chunks',
            )

        if not isinstance(chunks, list):
            raise HTTPException(
                status_code=400,
                detail='"chunks" must be a list',
            )
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise HTTPException(
                    status_code=400, detail="Each chunk must be a dictionary"
                )
            if "H" not in chunk or "T" not in chunk:
                raise HTTPException(
                    status_code=400,
                    detail='Each chunk must have "H" and "T" fields',
                )

    workload_definition["days"] = days

    # Create and save the composite workload
    workload = Composite.from_dict(workload_definition)
    workload.save()
    return name


class TailPerfRequest(BaseModel):
    workload_name: str
    blueprint_name: str
    query_router_name: str
    percentiles: list[int]


@router.post("/composite/tail_perf", response_model=dict)
def get_composite_workload_tail_performance(payload: TailPerfRequest):
    """
    For the named composite workload, return the performance at the specified
    percentile on the specified blueprint.

    Parameters (JSON body):
        workload_name: The name of the composite workload.
        blueprint_name: The name of the blueprint.
        query_router_name: The name of the query router.
        percentiles: A list of percentiles to compute performance for.

    Returns:
        A dictionary where each key is a percentile and each value is a list,
            containing the performance at the specified percentile for each day
            of the workload.
    """
    workload = Composite.load(payload.workload_name)
    blueprint = Blueprint.from_config(payload.blueprint_name)
    query_router = QueryRouter.from_name(
        payload.query_router_name, blueprint=blueprint
    )
    d: dict[int, list[float]] = {p: [] for p in payload.percentiles}
    for day_idx in range(len(workload.days)):

        perf: SLOStrategyPerformance = SLOStrategy.evaluate_suggestion(
            workload=workload,
            day_idx=day_idx,
            latency_slo_s=0,
            blueprint=blueprint,
            query_router=query_router,
        )

        for p in payload.percentiles:
            d[p].append(perf.latency_s_at_quantile(p / 100.0))
    return d


@router.get("/composite/{name}/blueprints_and_routers", response_model=dict)
def get_composite_workload_blueprints_and_routers(name: str):
    """
    For the composite workload, return a dictionary of blueprints and routers
    that have been used to run the workload.

    Parameters:
        name: The name of the composite workload.

    Returns:
        A dictionary with keys "blueprints" and "routers", each mapping to a list
        of names.
    """
    workload = Composite.load(name)
    return workload.get_available_blueprints_and_query_routers()


@router.get("/composite/{name}/definition_image")
def get_composite_workload_definition_image(name: str):
    base = pu.get_data_path()
    image_path = os.path.join(
        base, "composite_workloads", name, f"{name}_definition.png"
    )
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404, detail="Definition image not found"
        )
    return FileResponse(image_path, media_type="image/png")
