"""
Shared pytest fixtures for the Quantis test suite.
"""

from typing import Any

import pytest


@pytest.fixture
def test_client() -> Any:
    from api.app import app
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def sample_model() -> Any:
    from models.train_model import TemporalFusionTransformer

    model = TemporalFusionTransformer(input_size=128)
    return model


@pytest.fixture
def sample_data() -> Any:
    return {"features": [0.1] * 128, "api_key": "test_key"}


@pytest.fixture
def mock_mlflow() -> Any:
    class MockMLflow:
        def __init__(self):
            self.metrics = {}
            self.params = {}

        def log_metric(self, key, value):
            self.metrics[key] = value

        def log_param(self, key, value):
            self.params[key] = value

    return MockMLflow()
