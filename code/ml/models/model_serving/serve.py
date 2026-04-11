"""
Model serving utilities for Quantis.
Wraps trained models for MLflow pyfunc deployment.
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


class ModelWrapper:
    """
    MLflow pyfunc-compatible wrapper for PyTorch models.
    """

    def load_context(self, context: Any) -> None:
        try:
            import torch

            self.model = torch.load(context.artifacts["model_path"], map_location="cpu")
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def predict(self, context: Any, model_input: Any) -> Any:
        import torch

        input_tensor = torch.tensor(model_input.values, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.numpy()


@router.post("/model/deploy")
async def deploy_model(model_registry_path: str) -> dict:
    """
    Save a model for serving via MLflow pyfunc.

    Args:
        model_registry_path: Local path or MLflow URI to the model artifact.

    Returns:
        Status dict.
    """
    try:
        import mlflow.pyfunc
    except ImportError:
        return {"status": "error", "detail": "mlflow not installed"}

    try:
        mlflow.pyfunc.save_model(
            path="model_serving",
            python_model=ModelWrapper(),
            artifacts={"model_path": model_registry_path},
        )
        return {"status": "deployed", "path": model_registry_path}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
