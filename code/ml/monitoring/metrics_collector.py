"""
Metrics collection utilities for Quantis monitoring.
Supports AWS CloudWatch as a backend.
"""

from typing import Any

VALID_METRICS = {"PredictionLatency", "ModelAccuracy", "RequestCount", "ErrorRate"}


class MetricsCollector:
    """
    Collects and publishes operational metrics to AWS CloudWatch.
    """

    def __init__(self, namespace: str = "Quantis/Model") -> None:
        self.namespace = namespace
        self._metrics: list = []

    def record_metric(self, metric_name: str, value: float) -> Any:
        """
        Record a single metric value and publish it to CloudWatch.

        Args:
            metric_name: Name of the metric (must be non-empty and valid).
            value: Numeric value (must be >= 0).

        Raises:
            ValueError: If metric_name is empty or value is negative.
        """
        if not metric_name:
            raise ValueError("metric_name must be a non-empty string.")
        if value < 0:
            raise ValueError(
                f"Metric value must be >= 0, got {value} for metric '{metric_name}'."
            )

        import boto3

        cloudwatch = boto3.client("cloudwatch")
        cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": "None",
                }
            ],
        )
        self._metrics.append({"name": metric_name, "value": value})

    def get_recorded_metrics(self) -> list:
        """Return all metrics recorded in this session."""
        return list(self._metrics)
