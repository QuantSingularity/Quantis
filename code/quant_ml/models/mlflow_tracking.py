"""
MLflow experiment tracking utilities for Quantis.

Fixes vs original:
- log_experiment now checks for an active run before starting a new one
  (prevents MlflowException: "Run ... is already active" on nested calls)
- register_model guards against None model_uri more explicitly
- log_metrics gracefully handles both tracker objects and bare MLflow context
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_experiment(params: Any, metrics: Any, model: Any) -> None:
    """Log experiment parameters, metrics, and model artifact to MLflow."""
    if not MLFLOW_AVAILABLE:
        logger.debug("mlflow not installed; skipping experiment logging.")
        return

    # BUG FIX: calling mlflow.start_run() when a run is already active raises
    # MlflowException.  Use nested=True so this always works safely.
    with mlflow.start_run(nested=bool(mlflow.active_run())):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        try:
            mlflow.pytorch.log_model(model, "model")
        except Exception as e:
            logger.debug("pytorch model logging skipped: %s", e)
        try:
            mlflow.log_artifact("data/processed/feature_map.json")
        except Exception:
            pass  # artifact may not exist in all environments


def register_model(model_name: str, run_id: Optional[str]) -> None:
    """Register a model in the MLflow model registry."""
    if not MLFLOW_AVAILABLE:
        return
    if not run_id:
        logger.warning("register_model called with empty run_id — skipping.")
        return
    model_uri = f"runs:/{run_id}/model"
    try:
        mlflow.register_model(model_uri, model_name)
    except Exception as e:
        logger.warning("Model registration failed: %s", e)


def log_metrics(metrics: Dict[str, float], tracker: Any = None) -> None:
    """
    Log metrics to either a provided tracker object or MLflow.
    Supports mock trackers used in tests (with a log_metric method).
    """
    if tracker is not None:
        for key, value in metrics.items():
            tracker.log_metric(key, value)
        return
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.debug("mlflow.log_metrics failed: %s", e)
