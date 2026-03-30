from typing import Any
from unittest.mock import MagicMock, patch
import pytest
from api.app import app
from api.middleware.auth import validate_api_key
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def test_client() -> Any:
    return TestClient(app)


@pytest.fixture
def sample_data() -> Any:
    return {"features": [0.1] * 128, "api_key": "test_key"}


@pytest.fixture
def mock_env_api_key(monkeypatch: Any) -> Any:
    monkeypatch.setenv("API_SECRET", "test_key")


def test_predict_with_different_feature_lengths(
    test_client: Any, mock_env_api_key: Any
) -> Any:
    min_data = {"features": [0.1] * 10, "api_key": "test_key"}
    response = test_client.post("/predict", json=min_data)
    assert response.status_code == 422
    max_data = {"features": [0.1] * 256, "api_key": "test_key"}
    response = test_client.post("/predict", json=max_data)
    assert response.status_code == 422
    correct_data = {"features": [0.1] * 128, "api_key": "test_key"}
    response = test_client.post("/predict", json=correct_data)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_with_extreme_values(test_client: Any, mock_env_api_key: Any) -> Any:
    large_values = {"features": [10000000000.0] * 128, "api_key": "test_key"}
    response = test_client.post("/predict", json=large_values)
    assert response.status_code == 200
    small_values = {"features": [1e-10] * 128, "api_key": "test_key"}
    response = test_client.post("/predict", json=small_values)
    assert response.status_code == 200
    mixed_values = {
        "features": [-0.5 if i % 2 == 0 else 0.5 for i in range(128)],
        "api_key": "test_key",
    }
    response = test_client.post("/predict", json=mixed_values)
    assert response.status_code == 200


def test_predict_missing_api_key_variations(test_client: Any) -> Any:
    data_without_key = {"features": [0.1] * 128}
    response = test_client.post("/predict", json=data_without_key)
    assert response.status_code == 422
    data_empty_key = {"features": [0.1] * 128, "api_key": ""}
    response = test_client.post("/predict", json=data_empty_key)
    assert response.status_code == 403
    data_without_key = {"features": [0.1] * 128}
    response = test_client.post(
        "/predict", json=data_without_key, headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code in [401, 403, 422]


@patch("api.endpoints.prediction.joblib.load")
def test_model_loading_behavior(
    mock_load: Any, test_client: Any, mock_env_api_key: Any, sample_data: Any
) -> Any:
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.1, 0.2, 0.3]]
    mock_model.predict_proba.return_value = [[0.2, 0.5, 0.3]]
    mock_load.return_value = mock_model
    response = test_client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    mock_load.side_effect = FileNotFoundError("Model file not found")
    response = test_client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_model_health_detailed(test_client: Any) -> Any:
    response = test_client.get("/model_health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "version" in response.json()
    response = test_client.get("/model_health", headers={"Accept": "application/xml"})
    assert response.status_code == 406
    response = test_client.post("/model_health")
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_api_key_validation_with_env_vars(monkeypatch):
    monkeypatch.setenv("API_SECRET", "correct_key")
    result = await validate_api_key("correct_key")
    assert result == "correct_key"
    with pytest.raises(HTTPException) as excinfo:
        await validate_api_key("wrong_key")
    assert excinfo.value.status_code == 403
    monkeypatch.delenv("API_SECRET")
    with pytest.raises(HTTPException) as excinfo:
        await validate_api_key("any_key")
    assert excinfo.value.status_code == 403


def test_prediction_response_format(
    test_client: Any, mock_env_api_key: Any, sample_data: Any
) -> Any:
    response = test_client.post("/predict", json=sample_data)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert "confidence" in response_data
    assert isinstance(response_data["prediction"], list)
    assert isinstance(response_data["confidence"], (int, float))
    assert 0 <= response_data["confidence"] <= 1


def test_predict_with_malformed_features(
    test_client: Any, mock_env_api_key: Any
) -> Any:
    non_numeric = {"features": ["a", "b", "c"] + [0.1] * 125, "api_key": "test_key"}
    response = test_client.post("/predict", json=non_numeric)
    assert response.status_code == 422
    none_values = {"features": [None] * 128, "api_key": "test_key"}
    response = test_client.post("/predict", json=none_values)
    assert response.status_code == 422
    mixed_types = {
        "features": [0.1, "string", 1, True] + [0.1] * 124,
        "api_key": "test_key",
    }
    response = test_client.post("/predict", json=mixed_types)
    assert response.status_code == 422


def test_predict_with_different_http_methods(
    test_client: Any, mock_env_api_key: Any, sample_data: Any
) -> Any:
    response = test_client.get("/predict", params=sample_data)
    assert response.status_code == 405
    response = test_client.put("/predict", json=sample_data)
    assert response.status_code == 405
    response = test_client.delete("/predict")
    assert response.status_code == 405
