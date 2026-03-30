from typing import Any
import pytest
from api.app import app
from api.middleware.auth import ApiKeyManager, RateLimiter, Roles
from fastapi.testclient import TestClient


@pytest.fixture
def test_client() -> Any:
    return TestClient(app)


@pytest.fixture
def mock_env_api_key(monkeypatch: Any) -> Any:
    monkeypatch.setenv("API_SECRET", "test_key")
    monkeypatch.setenv("JWT_SECRET", "test_jwt_secret")


@pytest.fixture
def admin_api_key() -> Any:
    return ApiKeyManager.create_api_key(
        user_id="test_admin", role=Roles.ADMIN, expiry_days=1
    )


@pytest.fixture
def user_api_key() -> Any:
    return ApiKeyManager.create_api_key(
        user_id="test_user", role=Roles.USER, expiry_days=1
    )


@pytest.fixture
def readonly_api_key() -> Any:
    return ApiKeyManager.create_api_key(
        user_id="test_readonly", role=Roles.READONLY, expiry_days=1
    )


def test_api_key_creation() -> Any:
    api_key = ApiKeyManager.create_api_key(
        user_id="test_user", role=Roles.USER, expiry_days=30
    )
    assert api_key is not None
    assert len(api_key) > 0
    key_info = ApiKeyManager.validate_api_key(api_key)
    assert key_info["user_id"] == "test_user"
    assert key_info["role"] == Roles.USER


def test_api_key_validation() -> Any:
    api_key = ApiKeyManager.create_api_key(user_id="test_validation", role=Roles.ADMIN)
    key_info = ApiKeyManager.validate_api_key(api_key)
    assert key_info["user_id"] == "test_validation"
    assert key_info["role"] == Roles.ADMIN
    with pytest.raises(Exception):
        ApiKeyManager.validate_api_key("invalid_key")


def test_api_key_revocation() -> Any:
    api_key = ApiKeyManager.create_api_key(user_id="test_revocation", role=Roles.USER)
    key_info = ApiKeyManager.validate_api_key(api_key)
    assert key_info["user_id"] == "test_revocation"
    success = ApiKeyManager.revoke_api_key(api_key)
    assert success is True
    with pytest.raises(Exception):
        ApiKeyManager.validate_api_key(api_key)


def test_rate_limiter() -> Any:
    limiter = RateLimiter(requests_per_minute=3)
    user = {"user_id": "test_rate_limit"}
    for _ in range(3):
        result = limiter(user)
        assert result == user
    with pytest.raises(Exception):
        limiter(user)


def test_predict_endpoint_with_roles(
    test_client: Any, user_api_key: Any, readonly_api_key: Any
) -> Any:
    response = test_client.post(
        "/predict", json={"features": [0.1] * 128}, headers={"X-API-Key": user_api_key}
    )
    assert response.status_code == 200
    response = test_client.post(
        "/predict",
        json={"features": [0.1] * 128},
        headers={"X-API-Key": readonly_api_key},
    )
    assert response.status_code == 403


def test_model_health_with_roles(
    test_client: Any, user_api_key: Any, readonly_api_key: Any
) -> Any:
    response = test_client.get("/model_health", headers={"X-API-Key": user_api_key})
    assert response.status_code == 200
    response = test_client.get("/model_health", headers={"X-API-Key": readonly_api_key})
    assert response.status_code == 200


def test_user_management_endpoints(
    test_client: Any, admin_api_key: Any, user_api_key: Any
) -> Any:
    response = test_client.post(
        "/users",
        json={
            "username": "newuser",
            "email": "newuser@example.com",
            "role": "user",
            "password": "password123",
        },
        headers={"X-API-Key": admin_api_key},
    )
    assert response.status_code == 200
    response = test_client.post(
        "/users",
        json={
            "username": "newuser2",
            "email": "newuser2@example.com",
            "role": "user",
            "password": "password123",
        },
        headers={"X-API-Key": user_api_key},
    )
    assert response.status_code == 403


def test_api_key_management_endpoints(
    test_client: Any, admin_api_key: Any, user_api_key: Any
) -> Any:
    response = test_client.post(
        "/api-keys",
        json={"user_id": "newuser", "role": "user", "expiry_days": 30},
        headers={"X-API-Key": admin_api_key},
    )
    assert response.status_code == 200
    assert "api_key" in response.json()
    response = test_client.post(
        "/api-keys",
        json={"user_id": "newuser", "role": "user", "expiry_days": 30},
        headers={"X-API-Key": user_api_key},
    )
    assert response.status_code == 403


def test_current_user_endpoint(test_client: Any, user_api_key: Any) -> Any:
    response = test_client.get("/users/me", headers={"X-API-Key": user_api_key})
    assert response.status_code == 200
    assert response.json()["user_id"] == "test_user"
    assert response.json()["role"] == Roles.USER
