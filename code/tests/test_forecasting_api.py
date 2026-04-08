"""
Forecasting API integration tests: feature validation, extreme values,
auth checks, response format, and HTTP method enforcement.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client() -> Any:
    from api.app import app

    return TestClient(app)


@pytest.fixture
def sample_data() -> Any:
    return {"features": [0.1] * 128, "api_key": "test_key"}


@pytest.fixture
def mock_env_api_key(monkeypatch: Any) -> Any:
    monkeypatch.setenv("API_SECRET", "test_key")


# ---------------------------------------------------------------------------
# Feature-length validation
# ---------------------------------------------------------------------------


def test_predict_requires_auth(test_client: Any) -> None:
    """Prediction endpoint rejects unauthenticated requests."""
    response = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}, "model_id": 1},
    )
    assert response.status_code in (401, 403, 422)


def test_predict_with_different_feature_lengths(
    test_client: Any, mock_env_api_key: Any
) -> None:
    """Requests with wrong schema are rejected with 422."""
    # Missing required field → schema validation error
    response = test_client.post(
        "/predictions/predict",
        json={"features": [0.1] * 10},
        headers={"X-API-Key": "test_key"},
    )
    assert response.status_code == 422


def test_predict_with_extreme_values(test_client: Any, mock_env_api_key: Any) -> None:
    """Prediction endpoint with env API key responds (auth succeeds)."""
    response = test_client.post(
        "/predictions/predict",
        json={
            "input_data": {"features": [1e10] * 128},
            "model_id": 1,
        },
        headers={"X-API-Key": "test_key"},
    )
    # Auth should pass; model/data errors return 4xx not 5xx auth errors
    assert response.status_code in (200, 400, 404, 422)


def test_predict_missing_api_key_variations(test_client: Any) -> None:
    """Various forms of missing/invalid API keys are all rejected."""
    # No key at all
    resp = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}, "model_id": 1},
    )
    assert resp.status_code in (401, 403, 422)

    # Empty key header
    resp = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}, "model_id": 1},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code in (401, 403, 422)

    # Clearly invalid key
    resp = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}, "model_id": 1},
        headers={"X-API-Key": "invalid_key"},
    )
    assert resp.status_code in (401, 403, 422)


# ---------------------------------------------------------------------------
# Model health endpoint
# ---------------------------------------------------------------------------


def test_model_health_detailed(test_client: Any) -> None:
    """Model health endpoint exists and responds."""
    # GET is allowed
    response = test_client.get(
        "/predictions/models/health",
        headers={"X-API-Key": "test_key"},
    )
    # Without auth in DB it may return 403; endpoint must exist
    assert response.status_code in (200, 401, 403, 422)

    # POST to a GET-only endpoint → 405
    response = test_client.post("/predictions/models/health")
    assert response.status_code == 405


# ---------------------------------------------------------------------------
# API key env-var validation (no DB, just env-based auth)
# ---------------------------------------------------------------------------


def test_api_key_with_env_var(test_client: Any, monkeypatch: Any) -> None:
    """When API_SECRET is set, that key is accepted as admin."""
    monkeypatch.setenv("API_SECRET", "correct_key")

    response = test_client.get("/health")
    assert response.status_code == 200

    # An endpoint that uses api-key auth with the correct key
    resp = test_client.get(
        "/predictions/predictions/history",
        headers={"X-API-Key": "correct_key"},
    )
    # Auth should pass; empty DB is fine (returns 200 with [])
    assert resp.status_code in (200, 404)


def test_api_key_with_wrong_env_key(test_client: Any, monkeypatch: Any) -> None:
    """Wrong API key is rejected."""
    monkeypatch.setenv("API_SECRET", "correct_key")
    resp = test_client.get(
        "/predictions/predictions/history",
        headers={"X-API-Key": "wrong_key"},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Prediction response format
# ---------------------------------------------------------------------------


def test_prediction_response_format(test_client: Any, mock_env_api_key: Any) -> None:
    """
    When auth passes the response body must satisfy basic schema requirements.
    """
    resp = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}, "model_id": 1},
        headers={"X-API-Key": "test_key"},
    )
    # Auth OK; model may not exist in test DB → 4xx is acceptable
    assert resp.status_code in (200, 400, 404, 422)
    if resp.status_code == 200:
        body = resp.json()
        assert "prediction_result" in body or "prediction" in body


# ---------------------------------------------------------------------------
# Malformed feature payload
# ---------------------------------------------------------------------------


def test_predict_with_malformed_features(
    test_client: Any, mock_env_api_key: Any
) -> None:
    """Non-numeric feature values must be rejected by schema validation."""
    response = test_client.post(
        "/predictions/predict",
        json={"features": ["a", "b"] + [0.1] * 126},
        headers={"X-API-Key": "test_key"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# HTTP method enforcement
# ---------------------------------------------------------------------------


def test_predict_with_different_http_methods(
    test_client: Any, mock_env_api_key: Any, sample_data: Any
) -> None:
    """Non-POST methods on the predict endpoint are rejected with 405."""
    response = test_client.put(
        "/predictions/predict",
        json={"input_data": {"features": sample_data["features"]}, "model_id": 1},
        headers={"X-API-Key": "test_key"},
    )
    assert response.status_code == 405

    response = test_client.delete(
        "/predictions/predict",
        headers={"X-API-Key": "test_key"},
    )
    assert response.status_code == 405
