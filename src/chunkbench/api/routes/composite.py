from fastapi import APIRouter, HTTPException
from typing import List
import os
import yaml
import json
from fastapi import Body

from chunkbench.building_blocks.composite import Composite

import chunkbench.path_utils as pu

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
                    status_code=400, detail='Each chunk must have "H" and "T" fields'
                )


    workload_definition["days"] = days

    # Create and save the composite workload
    workload = Composite.from_dict(workload_definition)
    workload.save()
    return name