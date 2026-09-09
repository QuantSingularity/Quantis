"""
Authentication and authorization tests: API key lifecycle, role-based access,
JWT handling, and rate limiting.
"""

from typing import Any

import pytest
from backend.middleware.auth import RateLimiter, Roles

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rate_limiter() -> RateLimiter:
    return RateLimiter(requests_per_minute=3)


# ---------------------------------------------------------------------------
# RateLimiter tests (in-process, no DB needed)
# ---------------------------------------------------------------------------


def test_rate_limiter(rate_limiter: RateLimiter) -> None:
    """RateLimiter allows requests up to the limit then raises."""
    from fastapi import HTTPException

    fake_user = {"user_id": "rate_test_user", "role": Roles.USER}

    # First 3 calls should succeed
    for _ in range(3):
        result = rate_limiter(fake_user)
        assert result == fake_user

    # 4th call should be rejected
    with pytest.raises(HTTPException) as exc_info:
        rate_limiter(fake_user)
    assert exc_info.value.status_code == 429


def test_rate_limiter_admin_gets_double_limit() -> None:
    """Admin users get 2× the base rate limit."""
    from fastapi import HTTPException

    limiter = RateLimiter(requests_per_minute=5)
    admin_user = {"user_id": "admin_rate_user", "role": Roles.ADMIN}

    # 10 calls should all succeed (5 * 2 = 10)
    for _ in range(10):
        result = limiter(admin_user)
        assert result == admin_user

    # 11th should be rejected
    with pytest.raises(HTTPException):
        limiter(admin_user)


def test_rate_limiter_per_user_isolation() -> None:
    """Rate limits are tracked independently per user."""
    from fastapi import HTTPException

    limiter = RateLimiter(requests_per_minute=2)
    user_a = {"user_id": "user_a", "role": Roles.USER}
    user_b = {"user_id": "user_b", "role": Roles.USER}

    limiter(user_a)
    limiter(user_a)

    # user_a is exhausted
    with pytest.raises(HTTPException):
        limiter(user_a)

    # user_b is unaffected
    result = limiter(user_b)
    assert result == user_b


# ---------------------------------------------------------------------------
# Roles constants
# ---------------------------------------------------------------------------


def test_roles_constants() -> None:
    assert Roles.ADMIN == "admin"
    assert Roles.USER == "user"
    assert Roles.READONLY == "readonly"


# ---------------------------------------------------------------------------
# JWT token creation / verification (no DB)
# ---------------------------------------------------------------------------


def test_jwt_create_and_verify() -> None:
    """JWT tokens can be created and their payload recovered."""
    from backend.middleware.auth import create_jwt_token, decode_jwt_token

    payload = {"user_id": 42, "username": "testuser"}
    token = create_jwt_token(payload)
    assert token is not None
    assert len(token) > 0

    decoded = decode_jwt_token(token)
    assert decoded is not None
    assert decoded["user_id"] == 42
    assert decoded["username"] == "testuser"


def test_jwt_invalid_token_returns_none() -> None:
    """Decoding a tampered/invalid token returns None instead of raising."""
    from backend.middleware.auth import decode_jwt_token

    result = decode_jwt_token("not.a.valid.token")
    assert result is None


def test_jwt_expired_token() -> None:
    """An expired token is rejected."""
    from datetime import timedelta

    from backend.middleware.auth import create_jwt_token, decode_jwt_token

    token = create_jwt_token({"user_id": 1}, expires_delta=timedelta(seconds=-1))
    result = decode_jwt_token(token)
    assert result is None


# ---------------------------------------------------------------------------
# Health endpoint (integration, no auth required)
# ---------------------------------------------------------------------------


def test_health_endpoint_no_auth(test_client: Any) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_root_endpoint(test_client: Any) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
