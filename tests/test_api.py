"""Integration tests for Edge Model Quantization API."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine_ready"] is True
    assert data["target_bitwidth"] == 8


def test_quantize_tensor_endpoint(client):
    payload = {
        "matrix_rows": 128,
        "matrix_cols": 128,
        "seed": 42
    }
    response = client.post("/quantize/tensor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["matrix_shape"] == [128, 128]
    assert data["compression_ratio"] >= 3.8
    assert data["cosine_similarity"] > 0.99


def test_benchmark_suite_endpoint(client):
    response = client.get("/benchmark/suite")
    assert response.status_code == 200
    data = response.json()
    assert len(data["profiles"]) == 3
    assert data["max_compression_ratio"] >= 4.0
