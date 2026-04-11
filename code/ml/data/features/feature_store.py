"""
Feature store integration using Feast.
Provides feature view definitions for the Quantis ML pipeline.
"""

from datetime import timedelta
from typing import Any


def create_feature_view(repo_path: str = ".") -> Any:
    """
    Create and apply feature views to the Feast feature store.

    Args:
        repo_path: Path to the Feast repository configuration.

    Returns:
        The applied FeatureView object, or None if Feast is not available.
    """
    try:
        from feast import Entity, FeatureStore, FeatureView, Field, ValueType
        from feast.types import Float32, Int64

        driver = Entity(
            name="driver_id",
            value_type=ValueType.INT64,
            description="Driver identifier",
        )

        fs = FeatureStore(repo_path=repo_path)

        feature_view = FeatureView(
            name="driver_stats",
            entities=["driver_id"],
            ttl=timedelta(days=365),
            schema=[
                Field(name="trips_today", dtype=Int64),
                Field(name="rating", dtype=Float32),
                Field(name="conv_rate", dtype=Float32),
                Field(name="acc_rate", dtype=Float32),
            ],
        )

        fs.apply([driver, feature_view])
        return feature_view

    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "feast package not installed; feature store operations are unavailable."
        )
        return None
