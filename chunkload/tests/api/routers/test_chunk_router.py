from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

from chunkload.api.routers.chunk_router import router
from chunkload.building_blocks.chunk import Chunk


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    Create a TestClient that mounts the chunk router so tests can call the
    endpoint under test without starting the full application.
    """
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as tc:
        yield tc


def test_get_chunk_graphics_returns_expected_mappings(client: TestClient) -> None:
    """
    Ensure GET /chunk/graphics returns the H_SHAPE_MAP and T_COLOR_MAP in a
    JSON-serializable form matching the Chunk class definitions.
    """
    resp = client.get("/chunk/graphics")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    assert "H_SHAPE_MAP" in body and "T_COLOR_MAP" in body

    # Build the expected structure with stringified keys (JSON object keys
    # are always strings) and then run through FastAPI's encoder.
    expected_raw = {
        "H_SHAPE_MAP": {str(k): v for k, v in Chunk.H_SHAPE_MAP.items()},
        "T_COLOR_MAP": {str(k): v for k, v in Chunk.T_COLOR_MAP.items()},
    }
    expected = jsonable_encoder(expected_raw)
    assert body == expected


def test_chunk_graphics_post_not_allowed(client: TestClient) -> None:
    """
    Verify that POST to /chunk/graphics is not allowed (method not
    implemented) and returns a 405 status code.
    """
    resp = client.post("/chunk/graphics", json={"unused": "data"})
    assert resp.status_code == 405


def test_chunk_graphics_idempotent_with_query_params(client: TestClient) -> None:
    """
    Confirm that adding arbitrary query parameters does not change the
    returned mappings (endpoint is read-only and deterministic).
    """
    r1 = client.get("/chunk/graphics")
    r2 = client.get("/chunk/graphics?foo=bar&n=1")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
