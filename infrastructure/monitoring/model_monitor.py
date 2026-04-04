import numpy as np
from typing import Any, Dict, List, Optional
from prometheus_client import Gauge, Counter, Histogram


class ModelMonitor:

    def __init__(self) -> None:
        self.data_drift = Gauge("data_drift", "KL Divergence of Input Data")
        self.concept_drift = Gauge("concept_drift", "Performance Degradation")
        self.prediction_latency = Histogram(
            "prediction_latency_seconds",
            "Model prediction latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        self.prediction_counter = Counter(
            "predictions_total", "Total number of predictions made"
        )
        self.feature_importance: Dict[str, float] = {}
        self._feature_gauges: Dict[str, Gauge] = {}

    def calculate_drift(self, reference: np.ndarray, production: np.ndarray) -> float:
        kl_div = self._kl_divergence(reference, production)
        self.data_drift.set(kl_div)
        return kl_div

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        p = p / p.sum()
        q = q / q.sum()

        epsilon = 1e-10
        p = np.clip(p, epsilon, None)
        q = np.clip(q, epsilon, None)

        return float(np.sum(p * np.log(p / q)))

    def track_feature_importance(self, shap_values: List[float]) -> None:
        for idx, value in enumerate(shap_values):
            feature_name = f"feature_{idx}"
            self.feature_importance[feature_name] = float(value)
            if feature_name not in self._feature_gauges:
                self._feature_gauges[feature_name] = Gauge(
                    f"feature_importance_{idx}",
                    f"SHAP importance for feature {idx}",
                )
            self._feature_gauges[feature_name].set(float(value))

    def record_prediction(self, latency_seconds: float) -> None:
        self.prediction_latency.observe(latency_seconds)
        self.prediction_counter.inc()

    def set_concept_drift(self, degradation: float) -> None:
        self.concept_drift.set(degradation)
