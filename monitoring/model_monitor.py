import numpy as np
from prometheus_client import Gauge


class ModelMonitor:

    def __init__(self) -> None:
        self.data_drift = Gauge("data_drift", "KL Divergence of Input Data")
        self.concept_drift = Gauge("concept_drift", "Performance Degradation")
        self.feature_importance = {}

    def calculate_drift(self, reference: Any, production: Any) -> Any:
        kl_div = self._kl_divergence(reference, production)
        self.data_drift.set(kl_div)

    def _kl_divergence(self, p: Any, q: Any) -> Any:
        return np.sum(p * np.log(p / q))

    def track_feature_importance(self, shap_values: Any) -> Any:
        for idx, value in enumerate(shap_values):
            self.feature_importance[f"feature_{idx}"] = value
