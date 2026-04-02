"""
Prediction endpoints with database integration
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.auth import (
    admin_required,
    prediction_rate_limit,
    readonly_or_above,
    user_or_admin_required,
    validate_api_key,
)
from ..schemas import PredictionCreate, PredictionResponse
from ..services.model_service import ModelService
from ..services.prediction_service import PredictionService

router = APIRouter()


# Pydantic models
class BatchPredictionCreate(BaseModel):
    model_id: int
    input_data_list: List[List[float]]


class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    total_predictions: int
    successful_predictions: int
    failed_predictions: int


class PredictionHistory(BaseModel):
    id: int
    model_id: int
    model_name: str
    input_data: List[float]
    prediction_result: List[float]
    confidence_score: float
    execution_time_ms: int
    created_at: str


class PredictionStats(BaseModel):
    total_predictions: int
    avg_confidence: float
    avg_execution_time_ms: float
    predictions_by_model: dict
    predictions_by_day: dict


# Prediction endpoints
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionCreate,
    current_user: dict = Depends(user_or_admin_required),
    _: dict = Depends(prediction_rate_limit),
    db: Session = Depends(get_db),
):
    """Generate predictions using a trained model."""
    try:
        prediction_service = PredictionService(db)

        # Extract model_id from request or default to 1
        model_id = getattr(request, "model_id", 1)

        prediction = prediction_service.create_prediction(
            user_id=current_user["user_id"],
            model_id=model_id,
            input_data=request.features,
        )

        return PredictionResponse(
            prediction=prediction.prediction_result,
            confidence=prediction.confidence_score,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict_with_model(
    model_id: int,
    features: List[float],
    current_user: dict = Depends(user_or_admin_required),
    _: dict = Depends(prediction_rate_limit),
    db: Session = Depends(get_db),
):
    """Generate prediction using a specific model."""
    try:
        prediction_service = PredictionService(db)

        prediction = prediction_service.create_prediction(
            user_id=current_user["user_id"],
            model_id=model_id,
            input_data=features,
        )

        return PredictionResponse(
            prediction=prediction.prediction_result,
            confidence=prediction.confidence_score,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionCreate,
    current_user: dict = Depends(user_or_admin_required),
    db: Session = Depends(get_db),
):
    """Generate multiple predictions in batch."""
    try:
        prediction_service = PredictionService(db)

        predictions = prediction_service.batch_predict(
            user_id=current_user["user_id"],
            model_id=request.model_id,
            input_data_list=request.input_data_list,
        )

        successful_predictions = len(predictions)
        failed_predictions = len(request.input_data_list) - successful_predictions

        results = [
            {
                "id": pred.id,
                "prediction": pred.prediction_result,
                "confidence": pred.confidence_score,
                "execution_time_ms": pred.execution_time_ms,
            }
            for pred in predictions
        ]

        return BatchPredictionResponse(
            predictions=results,
            total_predictions=len(request.input_data_list),
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/predictions/history", response_model=List[PredictionHistory])
async def get_prediction_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_id: Optional[int] = Query(None),
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Get prediction history for the current user."""
    prediction_service = PredictionService(db)
    model_service = ModelService(db)

    if model_id:
        predictions = prediction_service.get_predictions_by_model(model_id, skip, limit)
    else:
        predictions = prediction_service.get_predictions_by_user(
            current_user["user_id"], skip, limit
        )

    # Cache model names
    model_names = {}
    for pred in predictions:
        if pred.model_id not in model_names:
            model = model_service.get_model_by_id(pred.model_id)
            model_names[pred.model_id] = (
                model.name if model else f"Model {pred.model_id}"
            )

    return [
        PredictionHistory(
            id=pred.id,
            model_id=pred.model_id,
            model_name=model_names[pred.model_id],
            input_data=pred.input_data,
            prediction_result=pred.prediction_result,
            confidence_score=pred.confidence_score,
            execution_time_ms=pred.execution_time_ms,
            created_at=pred.created_at.isoformat(),
        )
        for pred in predictions
    ]


@router.get("/predictions/stats", response_model=PredictionStats)
async def get_prediction_statistics(
    model_id: Optional[int] = Query(None),
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Get prediction statistics (admin = global, users = own only)."""
    prediction_service = PredictionService(db)

    user_id = None if current_user["role"] == "admin" else current_user["user_id"]

    stats = prediction_service.get_prediction_statistics(
        user_id=user_id, model_id=model_id
    )

    return PredictionStats(**stats)


@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: int,
    current_user: dict = Depends(validate_api_key),
    db: Session = Depends(get_db),
):
    """Get details of a specific prediction."""
    prediction_service = PredictionService(db)
    model_service = ModelService(db)

    prediction = prediction_service.get_prediction_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Enforce access
    if (
        current_user["role"] != "admin"
        and prediction.user_id != current_user["user_id"]
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    model = model_service.get_model_by_id(prediction.model_id)

    return {
        "id": prediction.id,
        "model_id": prediction.model_id,
        "model_name": model.name if model else f"Model {prediction.model_id}",
        "input_data": prediction.input_data,
        "prediction_result": prediction.prediction_result,
        "confidence_score": prediction.confidence_score,
        "execution_time_ms": prediction.execution_time_ms,
        "created_at": prediction.created_at.isoformat(),
        "user_id": prediction.user_id,
    }


# Model health endpoints
@router.get("/models/{model_id}/health")
async def check_model_health(
    model_id: int,
    current_user: dict = Depends(readonly_or_above),
    db: Session = Depends(get_db),
):
    """Check health status of a specific model."""
    try:
        model_service = ModelService(db)
        model = model_service.get_model_by_id(model_id)

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if model.status != "trained":
            return ModelHealthResponse(status="unhealthy", version="N/A")

        trained_model = model_service.load_trained_model(model_id)
        if not trained_model:
            return ModelHealthResponse(status="unhealthy", version="N/A")

        # Run a simple test input
        import numpy as np

        test = np.random.rand(10).tolist()
        trained_model.predict([test])

        return ModelHealthResponse(status="healthy", version=f"v{model.id}")

    except Exception:
        return ModelHealthResponse(status="unhealthy", version="N/A")


@router.get("/models/health", response_model=List[dict])
async def check_all_models_health(
    current_user: dict = Depends(readonly_or_above), db: Session = Depends(get_db)
):
    """Check health of all models."""
    model_service = ModelService(db)
    all_models = model_service.get_all_models()

    health = []
    for model in all_models:
        try:
            if model.status != "trained":
                status = "unhealthy"
            else:
                trained_model = model_service.load_trained_model(model.id)
                status = "healthy" if trained_model else "unhealthy"
        except Exception:
            status = "unhealthy"

        health.append(
            {
                "model_id": model.id,
                "model_name": model.name,
                "status": status,
                "model_status": model.status,
                "last_trained": (
                    model.trained_at.isoformat() if model.trained_at else None
                ),
            }
        )

    return health


@router.get("/admin/predictions", response_model=List[PredictionHistory])
async def get_all_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(admin_required),
    db: Session = Depends(get_db),
):
    """Admin endpoint: Get all predictions."""
    prediction_service = PredictionService(db)
    model_service = ModelService(db)

    predictions = prediction_service.get_all_predictions(skip, limit)

    model_names = {}
    for pred in predictions:
        if pred.model_id not in model_names:
            model = model_service.get_model_by_id(pred.model_id)
            model_names[pred.model_id] = (
                model.name if model else f"Model {pred.model_id}"
            )

    return [
        PredictionHistory(
            id=pred.id,
            model_id=pred.model_id,
            model_name=model_names[pred.model_id],
            input_data=pred.input_data,
            prediction_result=pred.prediction_result,
            confidence_score=pred.confidence_score,
            execution_time_ms=pred.execution_time_ms,
            created_at=pred.created_at.isoformat(),
        )
        for pred in predictions
    ]
