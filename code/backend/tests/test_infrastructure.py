"""
Tests for AWS infrastructure utilities and metrics collection.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from ml.models.aws_deploy import deploy_to_aws
from ml.monitoring.metrics_collector import MetricsCollector


@pytest.fixture
def mock_s3() -> Any:
    with patch("boto3.client") as mock_client:
        mock_s3_instance = Mock()
        mock_client.return_value = mock_s3_instance
        yield mock_s3_instance


@pytest.fixture
def mock_cloudwatch() -> Any:
    with patch("boto3.client") as mock_client:
        mock_cw_instance = Mock()
        mock_client.return_value = mock_cw_instance
        yield mock_cw_instance


def test_aws_deployment(mock_s3: Any) -> Any:
    model_path = "test_model.pt"
    bucket_name = "test-bucket"
    deploy_to_aws(model_path, bucket_name)
    mock_s3.upload_file.assert_called_once_with(
        model_path, bucket_name, f"models/{model_path}"
    )


def test_metrics_collector(mock_cloudwatch: Any) -> Any:
    collector = MetricsCollector()
    collector.record_metric("PredictionLatency", 0.5)
    collector.record_metric("ModelAccuracy", 0.95)
    assert mock_cloudwatch.put_metric_data.call_count == 2


def test_error_handling(mock_s3: Any) -> Any:
    mock_s3.upload_file.side_effect = Exception("Invalid credentials")
    with pytest.raises(Exception) as exc_info:
        deploy_to_aws("test_model.pt", "test-bucket")
    assert "Invalid credentials" in str(exc_info.value)


def test_metrics_validation(mock_cloudwatch: Any) -> Any:
    collector = MetricsCollector()
    with pytest.raises(ValueError):
        collector.record_metric("InvalidMetric", -1)
    with pytest.raises(ValueError):
        collector.record_metric("", 0.5)
