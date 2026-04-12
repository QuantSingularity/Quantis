"""
Feature store integration using Feast.
Provides feature view definitions for the Quantis ML pipeline.

Fixes vs original:
- Entity now uses the modern Feast API (join_keys list) instead of the
  deprecated value_type/ValueType pattern (removed in Feast >= 0.30).
- FeatureStore.apply() now receives a flat list with both Entity and FeatureView.
- Added QuantisFeatureStore convenience class for Quantis-specific features.
"""

import logging
from datetime import timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_feature_view(repo_path: str = ".") -> Optional[Any]:
    """
    Create and apply feature views to the Feast feature store.

    Args:
        repo_path: Path to the Feast repository configuration.

    Returns:
        The applied FeatureView object, or None if Feast is not available.
    """
    try:
        from feast import Entity, FeatureStore, FeatureView, Field
        from feast.types import Float32, Int64
    except ImportError:
        logger.warning(
            "feast package not installed; feature store operations are unavailable."
        )
        return None

    try:
        # BUG FIX: Feast >= 0.30 deprecated value_type/ValueType.
        # Use join_keys (list of column name strings) instead.
        driver = Entity(
            name="driver_id",
            join_keys=["driver_id"],
            description="Driver identifier",
        )

        feature_view = FeatureView(
            name="driver_stats",
            entities=[driver],
            ttl=timedelta(days=365),
            schema=[
                Field(name="trips_today", dtype=Int64),
                Field(name="rating", dtype=Float32),
                Field(name="conv_rate", dtype=Float32),
                Field(name="acc_rate", dtype=Float32),
            ],
        )

        fs = FeatureStore(repo_path=repo_path)
        # BUG FIX: pass both entity and feature_view in the same apply() call
        fs.apply([driver, feature_view])
        return feature_view

    except Exception as e:
        logger.error("Failed to create/apply feature view: %s", e)
        return None


class QuantisFeatureStore:
    """
    Quantis-specific feature store wrapper.

    Provides helper methods to retrieve features for model inference
    using Feast's online store.
    """

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = repo_path
        self._store: Optional[Any] = None

    def _get_store(self) -> Any:
        if self._store is None:
            try:
                from feast import FeatureStore

                self._store = FeatureStore(repo_path=self.repo_path)
            except ImportError:
                raise ImportError(
                    "feast is required for QuantisFeatureStore. "
                    "Install with: pip install feast"
                )
        return self._store

    def get_online_features(self, entity_rows: list, feature_refs: list) -> dict:
        """
        Retrieve features from the online store for real-time inference.

        Args:
            entity_rows: List of dicts with entity key-value pairs.
            feature_refs: List of feature reference strings (e.g. "driver_stats:rating").

        Returns:
            Dict mapping feature names to lists of values.
        """
        store = self._get_store()
        response = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        return response.to_dict()
