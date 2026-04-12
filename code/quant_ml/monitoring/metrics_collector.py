"""
Metrics collection utilities for Quantis monitoring.
Supports AWS CloudWatch as a backend.

Fixes vs original:
- boto3 client is now cached at __init__ time (not re-created on every call)
- Added try/except around CloudWatch call so a missing IAM role doesn't crash the app
- Added VALID_METRICS guard (optional, warn-only to avoid blocking new metrics)
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

VALID_METRICS = {"PredictionLatency", "ModelAccuracy", "RequestCount", "ErrorRate"}


class MetricsCollector:
    """
    Collects and publishes operational metrics to AWS CloudWatch.
    Falls back to local-only logging if CloudWatch is unavailable.
    """

    def __init__(self, namespace: str = "Quantis/Model") -> None:
        self.namespace = namespace
        self._metrics: List[dict] = []
        self._cloudwatch: Optional[Any] = None
        self._cw_available: bool = True

        # BUG FIX: create the boto3 client once at init time, not on every record call.
        try:
            import boto3

            self._cloudwatch = boto3.client("cloudwatch")
        except ImportError:
            logger.warning("boto3 not installed; CloudWatch publishing is disabled.")
            self._cw_available = False
        except Exception as e:
            logger.warning(
                "CloudWatch client init failed (%s); metrics will be local-only.", e
            )
            self._cw_available = False

    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Record a single metric value and publish it to CloudWatch.

        Args:
            metric_name: Name of the metric (must be non-empty).
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

        if metric_name not in VALID_METRICS:
            logger.warning(
                "Unknown metric '%s' — recording anyway. Known metrics: %s",
                metric_name,
                VALID_METRICS,
            )

        self._metrics.append({"name": metric_name, "value": value})

        if self._cw_available and self._cloudwatch is not None:
            try:
                self._cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=[
                        {
                            "MetricName": metric_name,
                            "Value": value,
                            "Unit": "None",
                        }
                    ],
                )
            except Exception as e:
                # BUG FIX: don't let a transient AWS error propagate and kill the request
                logger.warning("CloudWatch put_metric_data failed: %s", e)

    def get_recorded_metrics(self) -> List[dict]:
        """Return all metrics recorded in this session."""
        return list(self._metrics)
