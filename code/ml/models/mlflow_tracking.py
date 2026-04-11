from typing import Any, Dict

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_experiment(params: Any, metrics: Any, model: Any) -> Any:
    """Log experiment parameters, metrics and model to MLflow."""
    if not MLFLOW_AVAILABLE:
        return
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        try:
            mlflow.pytorch.log_model(model, "model")
        except Exception:
            pass
        try:
            mlflow.log_artifact("data/processed/feature_map.json")
        except Exception:
            pass


def register_model(model_name: Any, run_id: Any) -> Any:
    """Register a model in the MLflow model registry."""
    if not MLFLOW_AVAILABLE or not run_id:
        return
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)


def log_metrics(metrics: Dict[str, float], tracker: Any = None) -> Any:
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
        except Exception:
            pass
