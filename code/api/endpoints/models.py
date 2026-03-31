"""
Model management endpoints
"""

from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.auth import (
    readonly_or_above,
    user_or_admin_required,
    validate_api_key,
)
from ..services.dataset_service import DatasetService
from ..services.model_service import ModelService
from ..services.user_service import UserService

router = APIRouter()


class ModelCreate(BaseModel):
    name: str
    description: str
    model_type: str
    dataset_id: int
    hyperparameters: Optional[dict] = {}


class ModelResponse(BaseModel):
    id: int
    name: str
    description: str
    model_type: str
    owner_id: int
    owner_username: str
    dataset_id: int
    dataset_name: str
    hyperparameters: dict
    metrics: Optional[dict]
    status: str
    file_path: Optional[str]
    created_at: str
    updated_at: Optional[str]
    trained_at: Optional[str]


class ModelTrainingRequest(BaseModel):
    hyperparameters: Optional[dict] = {}


class ModelMetrics(BaseModel):
    mse: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    r2_score: Optional[float]
    training_time: Optional[float]
    epochs: Optional[int]


@router.post("/models", response_model=ModelResponse)
async def create_model(
    model_data: ModelCreate,
    current_user: dict = Depends(user_or_admin_required),
    db: Session = Depends(get_db),
):
    """Create a new model."""
    model_service = ModelService(db)
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(model_data.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if current_user["role"] != "admin" and dataset.owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied to dataset")
    try:
        model = model_service.create_model(
            name=model_data.name,
            description=model_data.description,
            model_type=model_data.model_type,
            owner_id=current_user["user_id"],
            dataset_id=model_data.dataset_id,
            hyperparameters=model_data.hyperparameters,
        )
        return ModelResponse(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            owner_id=model.owner_id,
            owner_username=current_user["username"],
            dataset_id=model.dataset_id,
            dataset_name=dataset.name,
            hyperparameters=model.hyperparameters,
            metrics=model.metrics,
            status=model.status,
            file_path=model.file_path,
            created_at=model.created_at.isoformat(),
            updated_at=model.updated_at.isoformat() if model.updated_at else None,
            trained_at=model.trained_at.isoformat() if model.trained_at else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models", response_model=List[ModelResponse])
async def get_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    model_type: Optional[str] = Query(None),
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Get models for the current user."""
    model_service = ModelService(db)
    dataset_service = DatasetService(db)
    if current_user["role"] == "admin":
        models = model_service.get_all_models(skip, limit)
    else:
        models = model_service.get_models_by_owner(current_user["user_id"], skip, limit)
    if status:
        models = [m for m in models if m.status == status]
    if model_type:
        models = [m for m in models if m.model_type == model_type]
    user_service = UserService(db)
    result = []
    for model in models:
        owner = user_service.get_user_by_id(model.owner_id)
        dataset = dataset_service.get_dataset_by_id(model.dataset_id)
        result.append(
            ModelResponse(
                id=model.id,
                name=model.name,
                description=model.description,
                model_type=model.model_type,
                owner_id=model.owner_id,
                owner_username=owner.username if owner else "Unknown",
                dataset_id=model.dataset_id,
                dataset_name=dataset.name if dataset else "Unknown",
                hyperparameters=model.hyperparameters,
                metrics=model.metrics,
                status=model.status,
                file_path=model.file_path,
                created_at=model.created_at.isoformat(),
                updated_at=model.updated_at.isoformat() if model.updated_at else None,
                trained_at=model.trained_at.isoformat() if model.trained_at else None,
            )
        )
    return result


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: int,
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Get model by ID."""
    model_service = ModelService(db)
    dataset_service = DatasetService(db)
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if current_user["role"] != "admin" and model.owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    user_service = UserService(db)
    owner = user_service.get_user_by_id(model.owner_id)
    dataset = dataset_service.get_dataset_by_id(model.dataset_id)
    return ModelResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        owner_id=model.owner_id,
        owner_username=owner.username if owner else "Unknown",
        dataset_id=model.dataset_id,
        dataset_name=dataset.name if dataset else "Unknown",
        hyperparameters=model.hyperparameters,
        metrics=model.metrics,
        status=model.status,
        file_path=model.file_path,
        created_at=model.created_at.isoformat(),
        updated_at=model.updated_at.isoformat() if model.updated_at else None,
        trained_at=model.trained_at.isoformat() if model.trained_at else None,
    )


@router.put("/models/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelCreate,
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Update model information."""
    model_service = ModelService(db)
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if current_user["role"] != "admin" and model.owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    updated_model = model_service.update_model(
        model_id,
        name=model_update.name,
        description=model_update.description,
        model_type=model_update.model_type,
        hyperparameters=model_update.hyperparameters,
    )
    if not updated_model:
        raise HTTPException(status_code=500, detail="Failed to update model")
    dataset_service = DatasetService(db)
    dataset = dataset_service.get_dataset_by_id(updated_model.dataset_id)
    return ModelResponse(
        id=updated_model.id,
        name=updated_model.name,
        description=updated_model.description,
        model_type=updated_model.model_type,
        owner_id=updated_model.owner_id,
        owner_username=current_user["username"],
        dataset_id=updated_model.dataset_id,
        dataset_name=dataset.name if dataset else "Unknown",
        hyperparameters=updated_model.hyperparameters,
        metrics=updated_model.metrics,
        status=updated_model.status,
        file_path=updated_model.file_path,
        created_at=updated_model.created_at.isoformat(),
        updated_at=(
            updated_model.updated_at.isoformat() if updated_model.updated_at else None
        ),
        trained_at=(
            updated_model.trained_at.isoformat() if updated_model.trained_at else None
        ),
    )


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: int,
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Delete model."""
    model_service = ModelService(db)
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if current_user["role"] != "admin" and model.owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    success = model_service.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")
    return {"message": "Model deleted successfully"}


@router.post("/models/{model_id}/train")
async def train_model(
    model_id: int,
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(user_or_admin_required),
    db: Session = Depends(get_db),
):
    """Start model training."""
    model_service = ModelService(db)
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if current_user["role"] != "admin" and model.owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    if model.status == "training":
        raise HTTPException(status_code=400, detail="Model is already training")
    if training_request.hyperparameters:
        model_service.update_model(
            model_id, hyperparameters=training_request.hyperparameters
        )

    def train_model_task() -> Any:
        model_service.train_dummy_model(model_id)

    background_tasks.add_task(train_model_task)
    return {"message": "Model training started", "model_id": model_id}


@router.get("/models/{model_id}/training-status")
async def get_training_status(
    model_id: int,
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Get model training status."""
    model_service = ModelService(db)
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if current_user["role"] != "admin" and model.owner_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return {
        "model_id": model.id,
        "status": model.status,
        "metrics": model.metrics,
        "trained_at": model.trained_at.isoformat() if model.trained_at else None,
        "file_path": model.file_path,
    }


@router.get("/models/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    model_id: int,
    current_user: dict = Depends(readonly_or_above),
    db: Session = Depends(get_db),
):
    """Get model training metrics."""
    model_service = ModelService(db)
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if (
        current_user["role"] not in ["admin", "readonly"]
        and model.owner_id != current_user["user_id"]
    ):
        raise HTTPException(status_code=403, detail="Access denied")
    if not model.metrics:
        raise HTTPException(
            status_code=404, detail="No metrics available for this model"
        )
    return ModelMetrics(**model.metrics)


@router.get("/models/compare")
async def compare_models(
    model_ids: str = Query(..., description="Comma-separated list of model IDs"),
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Compare multiple models."""
    model_service = ModelService(db)
    try:
        model_id_list = [int(id.strip()) for id in model_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model IDs format")
    if len(model_id_list) > 10:
        raise HTTPException(
            status_code=400, detail="Cannot compare more than 10 models at once"
        )
    comparison_data = []
    for model_id in model_id_list:
        model = model_service.get_model_by_id(model_id)
        if not model:
            continue
        if (
            current_user["role"] not in ["admin", "readonly"]
            and model.owner_id != current_user["user_id"]
        ):
            continue
        comparison_data.append(
            {
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type,
                "status": model.status,
                "metrics": model.metrics,
                "trained_at": (
                    model.trained_at.isoformat() if model.trained_at else None
                ),
            }
        )
    return {"models": comparison_data, "comparison_count": len(comparison_data)}


@router.get("/models/types")
async def get_model_types(current_user: dict = Depends(readonly_or_above)):
    """Get available model types and their descriptions."""
    return {
        "model_types": [
            {
                "type": "tft",
                "name": "Temporal Fusion Transformer",
                "description": "Advanced transformer model for time series forecasting with attention mechanisms",
                "suitable_for": ["time_series", "forecasting", "multivariate"],
            },
            {
                "type": "lstm",
                "name": "Long Short-Term Memory",
                "description": "Recurrent neural network for sequence modeling and time series prediction",
                "suitable_for": ["time_series", "sequence_modeling", "forecasting"],
            },
            {
                "type": "arima",
                "name": "AutoRegressive Integrated Moving Average",
                "description": "Statistical model for time series analysis and forecasting",
                "suitable_for": ["time_series", "univariate", "statistical_analysis"],
            },
            {
                "type": "linear",
                "name": "Linear Regression",
                "description": "Simple linear model for regression tasks",
                "suitable_for": ["regression", "baseline", "interpretable"],
            },
            {
                "type": "random_forest",
                "name": "Random Forest",
                "description": "Ensemble method using multiple decision trees",
                "suitable_for": ["regression", "classification", "feature_importance"],
            },
            {
                "type": "xgboost",
                "name": "XGBoost",
                "description": "Gradient boosting framework for structured data",
                "suitable_for": ["regression", "classification", "high_performance"],
            },
        ]
    }
