import os
from typing import Generator

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import autoslo.utils.paths as pu
from autoslo.api.routers.composite_router import router
from autoslo.workload_definition.composite import Composite


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    Create a TestClient that mounts the composite router so tests can call
    endpoints without starting the full application.
    """
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as tc:
        yield tc


def test_list_and_get_composite_workloads(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """
    Test listing composites (empty vs present) and retrieving a composite
    definition via GET /composite/{name}.
    """
    monkeypatch.setattr(pu, "get_data_path", lambda: str(tmp_path))

    # Initially no composites
    resp = client.get("/composite")
    assert resp.status_code == 200
    assert resp.json() == []

    # Create a composite directory with a definition.yml and verify listing
    name = "my_comp"
    wd = os.path.join(str(tmp_path), "composite_workloads", name)
    os.makedirs(wd, exist_ok=True)
    definition = {"name": name, "monday_index": 0, "days": []}
    with open(os.path.join(wd, "definition.yml"), "w") as fh:
        yaml.dump(definition, fh, sort_keys=False)

    resp2 = client.get("/composite")
    assert resp2.status_code == 200
    assert name in resp2.json()

    # GET the composite definition
    resp3 = client.get(f"/composite/{name}")
    assert resp3.status_code == 200
    assert resp3.json() == definition

    # GET a non-existent composite -> 404
    resp4 = client.get("/composite/nonexistent")
    assert resp4.status_code == 404


def test_create_composite_workload_post_success(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """
    Test POST /composite/create succeeds for a valid workload definition.
    Patch Composite.save to avoid running the real save implementation.
    """
    monkeypatch.setattr(pu, "get_data_path", lambda: str(tmp_path))

    # Replace Composite.save with a simple implementation that writes the
    # definition.yml into the expected folder (avoids heavy internals).
    def fake_save(self: Composite) -> None:
        out_dir = os.path.join(str(tmp_path), "composite_workloads", self.name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "definition.yml"), "w") as fh:
            yaml.dump(self.to_dict(), fh, sort_keys=False)

    monkeypatch.setattr(Composite, "save", fake_save)

    payload = {
        "name": "created_comp",
        "monday_index": 1,
        "days": [
            {"chunks": [{"H": 10, "T": 30}]},
            {"chunks": [{"H": 25, "T": 60}]},
        ],
    }
    resp = client.post("/composite/create", json=payload)
    assert resp.status_code == 200
    assert resp.text.strip('"') == "created_comp"

    # Confirm definition file exists and contains correct name
    def_path = os.path.join(
        str(tmp_path), "composite_workloads", "created_comp", "definition.yml"
    )
    assert os.path.exists(def_path)
    with open(def_path, "r") as fh:
        loaded = yaml.safe_load(fh)
    assert loaded["name"] == "created_comp"


def test_create_composite_workload_post_bad_payloads(
    client: TestClient,
) -> None:
    """
    Verify POST /composite/create returns 400 for malformed payloads:
    - missing name
    - days not a list
    - chunk missing H/T
    """
    # missing name
    payload1 = {"monday_index": 0, "days": []}
    r1 = client.post("/composite/create", json=payload1)
    assert r1.status_code == 400

    # days not a list
    payload2 = {"name": "x", "days": "notalist"}
    r2 = client.post("/composite/create", json=payload2)
    assert r2.status_code == 400

    # chunk missing H/T
    payload3 = {"name": "x", "days": [{"chunks": [{"H": 5}]}]}
    r3 = client.post("/composite/create", json=payload3)
    assert r3.status_code == 400


def test_ground_truth_endpoint_uses_composite_method(
    client: TestClient, monkeypatch
) -> None:
    """
    Ensure POST /composite/ground_truth_smallest_adherent_endpoint delegates to
    Composite.ground_truth_smallest_adherent_endpoint and returns its value.
    """

    # Patch the Composite method to a stable deterministic function
    def fake_ground_truth(
        name: str, tail_slo_s: float, percentile: float = 95.0
    ):
        return [4, None]

    monkeypatch.setattr(
        Composite,
        "ground_truth_smallest_adherent_endpoint",
        staticmethod(fake_ground_truth),
    )

    resp = client.post(
        "/composite/ground_truth_smallest_adherent_endpoint",
        params={"workload_name": "any", "tail_slo_s": 1.0, "percentile": 95.0},
    )
    assert resp.status_code == 200
    assert resp.json() == [4, None]
