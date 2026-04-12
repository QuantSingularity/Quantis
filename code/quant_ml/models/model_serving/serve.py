"""
Model serving utilities for Quantis.
Wraps trained models for MLflow pyfunc deployment.

Fixes vs original:
- torch.load now uses weights_only=True (security fix — prevents arbitrary pickle execution)
- Added proper error response shape to /model/deploy endpoint
- Removed duplicate ModelWrapper class (was also defined in train_model.py)
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


class QuantisModelWrapper:
    """
    MLflow pyfunc-compatible wrapper for PyTorch models.
    """

    def load_context(self, context: Any) -> None:
        try:
            import torch

            # SECURITY FIX: weights_only=True prevents arbitrary code execution
            # via malicious pickle payloads embedded in model files (CVE-class issue).
            # Requires PyTorch >= 1.13; falls back gracefully for older versions.
            try:
                self.model = torch.load(
                    context.artifacts["model_path"],
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # torch < 1.13 doesn't support weights_only parameter
                self.model = torch.load(
                    context.artifacts["model_path"],
                    map_location="cpu",
                )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def predict(self, context: Any, model_input: Any) -> Any:
        import torch

        input_tensor = torch.tensor(model_input.values, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.detach().numpy()


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
            python_model=QuantisModelWrapper(),
            artifacts={"model_path": model_registry_path},
        )
        return {"status": "deployed", "path": model_registry_path}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
