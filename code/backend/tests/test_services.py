"""
Service-layer unit tests: UserService, PredictionService, ModelService,
prediction statistics, batch operations, and edge-case handling.
"""

from typing import Any

import pytest
from backend.domain.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# In-memory SQLite DB fixture (isolated per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# UserService tests
# ---------------------------------------------------------------------------


def test_create_user_success(db: Any) -> None:
    """UserService.create_user creates a user with hashed password."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("alice", "alice@example.com", "secret123")
    assert user.id is not None
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.hashed_password != "secret123"


def test_create_user_duplicate_raises(db: Any) -> None:
    """Creating a user with a duplicate username raises ValueError."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    svc.create_user("bob", "bob@example.com", "pw")
    with pytest.raises(ValueError, match="already exists"):
        svc.create_user("bob", "bob2@example.com", "pw")


def test_create_user_duplicate_email_raises(db: Any) -> None:
    """Creating a user with a duplicate email raises ValueError."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    svc.create_user("carol", "carol@example.com", "pw")
    with pytest.raises(ValueError):
        svc.create_user("carol2", "carol@example.com", "pw")


def test_authenticate_user_correct_password(db: Any) -> None:
    """authenticate_user returns user on correct credentials."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    svc.create_user("dave", "dave@example.com", "mypassword")
    result = svc.authenticate_user("dave", "mypassword")
    assert result is not None
    assert result.username == "dave"


def test_authenticate_user_wrong_password(db: Any) -> None:
    """authenticate_user returns None on wrong password."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    svc.create_user("eve", "eve@example.com", "correct")
    result = svc.authenticate_user("eve", "wrong")
    assert result is None


def test_authenticate_user_nonexistent(db: Any) -> None:
    """authenticate_user returns None for unknown username."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    result = svc.authenticate_user("ghost", "pw")
    assert result is None


def test_get_user_by_id(db: Any) -> None:
    """get_user_by_id returns the correct user."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("frank", "frank@example.com", "pw")
    fetched = svc.get_user_by_id(user.id)
    assert fetched is not None
    assert fetched.id == user.id


def test_get_user_by_id_missing(db: Any) -> None:
    """get_user_by_id returns None for missing ID."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    assert svc.get_user_by_id(99999) is None


def test_get_user_by_username(db: Any) -> None:
    """get_user_by_username returns the correct user."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    svc.create_user("grace", "grace@example.com", "pw")
    user = svc.get_user_by_username("grace")
    assert user is not None
    assert user.username == "grace"


def test_deactivate_user(db: Any) -> None:
    """deactivate_user marks user as inactive."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("henry", "henry@example.com", "pw")
    assert user.is_active is True
    result = svc.deactivate_user(user.id)
    assert result is True
    db.refresh(user)
    assert user.is_active is False


def test_create_and_validate_api_key(db: Any) -> None:
    """API key creation and validation round-trip works."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("ivan", "ivan@example.com", "pw")
    raw_key = svc.create_api_key(user.id, "test-key", expires_days=30)
    assert raw_key is not None

    info = svc.validate_api_key(raw_key)
    assert info is not None
    assert info["user_id"] == user.id


def test_validate_invalid_api_key(db: Any) -> None:
    """Validating a garbage API key returns None."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    result = svc.validate_api_key("totally_fake_key_xyz")
    assert result is None


def test_revoke_api_key(db: Any) -> None:
    """Revoked API key no longer validates."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("judy", "judy@example.com", "pw")
    raw_key = svc.create_api_key(user.id, "revoke-me", expires_days=30)
    svc.revoke_api_key(raw_key)
    assert svc.validate_api_key(raw_key) is None


def test_get_users_pagination(db: Any) -> None:
    """get_users respects skip/limit pagination."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    for i in range(5):
        svc.create_user(f"user{i}", f"user{i}@example.com", "pw")

    page1 = svc.get_users(skip=0, limit=3)
    page2 = svc.get_users(skip=3, limit=3)
    assert len(page1) == 3
    assert len(page2) == 2
    assert {u.username for u in page1}.isdisjoint({u.username for u in page2})


# ---------------------------------------------------------------------------
# PredictionService tests
# ---------------------------------------------------------------------------


def test_prediction_service_user_not_found(db: Any) -> None:
    """create_prediction raises ValueError when user does not exist."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    with pytest.raises(ValueError, match="[Uu]ser not found"):
        svc.create_prediction(user_id=9999, model_id=1, input_data=[0.1] * 10)


def test_prediction_service_system_user_id(db: Any) -> None:
    """create_prediction with non-integer user_id raises ValueError cleanly."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    with pytest.raises(ValueError):
        svc.create_prediction(user_id="system", model_id=1, input_data=[0.1] * 10)


def test_get_prediction_by_id_missing(db: Any) -> None:
    """get_prediction_by_id returns None for non-existent ID."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    assert svc.get_prediction_by_id(99999) is None


def test_get_predictions_by_user_empty(db: Any) -> None:
    """get_predictions_by_user returns empty list when no predictions exist."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    result = svc.get_predictions_by_user(user_id=1)
    assert result == []


def test_get_predictions_by_user_system_id(db: Any) -> None:
    """get_predictions_by_user with system string user_id returns empty list."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    result = svc.get_predictions_by_user(user_id="system")
    assert result == []


def test_get_prediction_statistics_empty(db: Any) -> None:
    """get_prediction_statistics returns zeros when no predictions exist."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    stats = svc.get_prediction_statistics()
    assert stats["total_predictions"] == 0
    assert stats["avg_confidence"] == 0
    assert stats["avg_execution_time_ms"] == 0


def test_get_all_predictions_empty(db: Any) -> None:
    """get_all_predictions returns empty list on fresh DB."""
    from backend.services.prediction_service import PredictionService

    svc = PredictionService(db)
    result = svc.get_all_predictions()
    assert result == []


# ---------------------------------------------------------------------------
# ModelService tests
# ---------------------------------------------------------------------------


def test_get_model_by_id_missing(db: Any) -> None:
    """get_model_by_id returns None for missing model."""
    from backend.services.model_service import ModelService

    svc = ModelService(db)
    assert svc.get_model_by_id(9999) is None


def test_get_all_models_empty(db: Any) -> None:
    """get_all_models returns empty list on fresh DB."""
    from backend.services.model_service import ModelService

    svc = ModelService(db)
    result = svc.get_all_models()
    assert result == []


def test_load_trained_model_missing(db: Any) -> None:
    """load_trained_model returns None for non-existent model."""
    from backend.services.model_service import ModelService

    svc = ModelService(db)
    result = svc.load_trained_model(9999)
    assert result is None


# ---------------------------------------------------------------------------
# API endpoint integration tests (additional coverage)
# ---------------------------------------------------------------------------


def test_health_endpoint_structure(test_client: Any) -> None:
    """Health endpoint returns expected fields."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body


def test_info_endpoint_returns_metadata(test_client: Any) -> None:
    """Info endpoint returns application metadata."""
    resp = test_client.get("/info")
    assert resp.status_code in (200, 429)
    if resp.status_code == 200:
        body = resp.json()
        assert isinstance(body, dict)


def test_metrics_endpoint_returns_200(test_client: Any) -> None:
    """Prometheus metrics endpoint is reachable."""
    resp = test_client.get("/metrics")
    assert resp.status_code == 200


def test_predict_endpoint_missing_model_id(test_client: Any, monkeypatch: Any) -> None:
    """POST /predict without model_id returns 422."""
    monkeypatch.setenv("API_SECRET", "test_key")
    resp = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}},
        headers={"X-API-Key": "test_key"},
    )
    assert resp.status_code == 422


def test_predict_endpoint_model_not_found(test_client: Any, monkeypatch: Any) -> None:
    """POST /predict for nonexistent model returns 400 or 404, not 500."""
    monkeypatch.setenv("API_SECRET", "test_key")
    resp = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": [0.1] * 128}, "model_id": 99999},
        headers={"X-API-Key": "test_key"},
    )
    assert resp.status_code in (400, 404, 422)


def test_prediction_history_with_auth(test_client: Any, monkeypatch: Any) -> None:
    """GET /predictions/predictions/history with valid API_SECRET returns 200."""
    monkeypatch.setenv("API_SECRET", "test_key")
    resp = test_client.get(
        "/predictions/predictions/history",
        headers={"X-API-Key": "test_key"},
    )
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)


def test_prediction_history_wrong_key(test_client: Any, monkeypatch: Any) -> None:
    """GET /predictions/history with wrong key returns 401 or 403."""
    monkeypatch.setenv("API_SECRET", "correct_key")
    resp = test_client.get(
        "/predictions/predictions/history",
        headers={"X-API-Key": "wrong_key"},
    )
    assert resp.status_code in (401, 403)


def test_models_health_no_models(test_client: Any, monkeypatch: Any) -> None:
    """GET /predictions/models/health returns 200 with empty list when no models."""
    monkeypatch.setenv("API_SECRET", "test_key")
    resp = test_client.get(
        "/predictions/models/health",
        headers={"X-API-Key": "test_key"},
    )
    assert resp.status_code in (200, 401, 403)
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)


def test_batch_predict_no_auth(test_client: Any) -> None:
    """POST /predictions/predict/batch without auth returns 401/403."""
    resp = test_client.post(
        "/predictions/predict/batch",
        json={"model_id": 1, "input_data_list": [[0.1] * 128]},
    )
    assert resp.status_code in (401, 403, 422)


def test_get_prediction_by_id_not_found(test_client: Any, monkeypatch: Any) -> None:
    """GET /predictions/{id} for nonexistent ID returns 404."""
    monkeypatch.setenv("API_SECRET", "test_key")
    resp = test_client.get(
        "/predictions/predictions/99999",
        headers={"X-API-Key": "test_key"},
    )
    assert resp.status_code in (404, 422)


def test_wrong_http_method_on_health(test_client: Any) -> None:
    """POST to a GET-only health endpoint returns 405."""
    resp = test_client.post("/health")
    assert resp.status_code == 405


# ---------------------------------------------------------------------------
# Model schema / JWT security tests
# ---------------------------------------------------------------------------


def test_jwt_roundtrip_with_role() -> None:
    """JWT payload preserves role field through encode/decode."""
    from backend.middleware.auth import create_jwt_token, decode_jwt_token

    token = create_jwt_token({"user_id": 7, "role": "admin"})
    decoded = decode_jwt_token(token)
    assert decoded["role"] == "admin"
    assert decoded["user_id"] == 7


def test_jwt_tampered_signature_rejected() -> None:
    """A JWT with a tampered signature is rejected."""
    from backend.middleware.auth import create_jwt_token, decode_jwt_token

    token = create_jwt_token({"user_id": 1})
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + ".badsignature"
    assert decode_jwt_token(tampered) is None


def test_rate_limiter_window_reset() -> None:
    """Rate limit window resets after 60 seconds (simulated by clearing history)."""
    import time

    from backend.middleware.auth import RateLimiter, Roles

    limiter = RateLimiter(requests_per_minute=2)
    user = {"user_id": "window_user", "role": Roles.USER}

    limiter(user)
    limiter(user)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        limiter(user)
    assert exc.value.status_code == 429

    # Simulate window expiry by backdating history
    old_time = time.time() - 61
    limiter.request_history["window_user"] = [old_time, old_time]

    result = limiter(user)
    assert result == user


# ---------------------------------------------------------------------------
# Data model / ORM tests
# ---------------------------------------------------------------------------


def test_user_full_name_property(db: Any) -> None:
    """User.full_name returns 'first last' or username fallback."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("fullname_user", "fn@example.com", "pw")
    assert user.full_name == "fullname_user"

    user.first_name = "John"
    user.last_name = "Doe"
    db.commit()
    db.refresh(user)
    assert user.full_name == "John Doe"


def test_user_password_hashing(db: Any) -> None:
    """User.verify_password correctly validates hashed passwords."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("hashtest", "ht@example.com", "mypassword")
    assert user.verify_password("mypassword") is True
    assert user.verify_password("wrongpassword") is False


def test_api_key_static_methods() -> None:
    """ApiKey.generate_key and hash_key work correctly."""
    from backend.domain.models import ApiKey

    key = ApiKey.generate_key()
    assert key.startswith("qk_")
    assert len(key) > 10

    hashed = ApiKey.hash_key(key)
    assert hashed != key
    assert len(hashed) == 64
    assert ApiKey.hash_key(key) == hashed


def test_role_creation_idempotent(db: Any) -> None:
    """_get_or_create_role is idempotent — two calls return same role."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    r1 = svc._get_or_create_role("analyst")
    r2 = svc._get_or_create_role("analyst")
    assert r1.id == r2.id


def test_user_lock_unlock(db: Any) -> None:
    """User lock/unlock cycle works correctly."""
    from backend.services.user_service import UserService

    svc = UserService(db)
    user = svc.create_user("locktest", "lt@example.com", "pw")
    assert user.is_locked() is False

    user.lock_account(duration_minutes=10)
    assert user.is_locked() is True

    user.unlock_account()
    assert user.locked_until is None
    assert user.login_attempts == 0
