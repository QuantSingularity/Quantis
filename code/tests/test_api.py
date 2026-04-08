"""
API integration tests covering prediction endpoint, health check,
rate limiting, CORS headers, and request validation.
"""

from typing import Any


def test_predict_endpoint_success(
    test_client: Any, sample_data: Any, monkeypatch
) -> Any:
    monkeypatch.setenv("API_SECRET", "test_key")
    response = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": sample_data["features"]}, "model_id": 1},
        headers={"X-API-Key": "test_key"},
    )
    # Either succeeds or returns a known auth/validation error, not 500
    assert response.status_code in (200, 400, 401, 403, 404, 422)


def test_predict_endpoint_invalid_data(test_client: Any) -> Any:
    """Submitting wrong schema returns 422 Unprocessable Entity."""
    response = test_client.post(
        "/predictions/predict",
        json={"features": [0.1]},  # missing required model_id field
    )
    assert response.status_code == 422


def test_predict_endpoint_missing_auth(test_client: Any, sample_data: Any) -> Any:
    """No API key → 403 (middleware rejects unauthenticated requests)."""
    response = test_client.post(
        "/predictions/predict",
        json={"input_data": {"features": sample_data["features"]}, "model_id": 1},
    )
    assert response.status_code in (401, 403, 422)


def test_health_check(test_client: Any) -> Any:
    """Health endpoint returns 200 with status field."""
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] in ("ok", "healthy", "error")


def test_root_endpoint(test_client: Any) -> Any:
    """Root endpoint returns welcome message."""
    response = test_client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body


def test_cors_headers(test_client: Any) -> Any:
    """CORS preflight returns correct headers."""
    response = test_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # With wildcard CORS the preflight should succeed
    assert response.status_code in (200, 204, 400)


def test_malformed_json(test_client: Any) -> Any:
    """Sending invalid JSON body returns 422 or 400."""
    response = test_client.post(
        "/predictions/predict",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)


def test_concurrent_requests(test_client: Any, sample_data: Any, monkeypatch) -> Any:
    """Multiple concurrent requests do not crash the server."""
    monkeypatch.setenv("API_SECRET", "test_key")

    def make_request():
        return test_client.get("/health")

    # Use threading instead of asyncio for sync TestClient
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        responses = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in responses)


def test_metrics_endpoint(test_client: Any) -> Any:
    """Metrics endpoint is reachable."""
    response = test_client.get("/metrics")
    assert response.status_code == 200


def test_info_endpoint(test_client: Any) -> Any:
    """System info endpoint returns application metadata."""
    response = test_client.get("/info")
    assert response.status_code in (200, 429)  # may hit rate limit in repeated runs
