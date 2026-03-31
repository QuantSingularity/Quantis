from datetime import timedelta
from typing import Any

from feast import Entity, FeatureStore, FeatureView, ValueType

driver = Entity(
    name="driver_id", value_type=ValueType.INT64, description="Driver identifier"
)
fs = FeatureStore(repo_path=".")


def create_feature_view() -> Any:
    fs.apply(
        [
            FeatureView(
                name="driver_stats",
                entities=["driver_id"],
                ttl=timedelta(days=365),
                features=[...],
            )
        ]
    )
