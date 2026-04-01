"""
Prediction service for handling model predictions
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import models
from .model_service import ModelService

logger = logging.getLogger(__name__)


class PredictionService:

    def __init__(self, db: Session) -> None:
        self.db = db
        self.model_service = ModelService(db)

    def create_prediction(
        self, user_id: int, model_id: int, input_data: List[float]
    ) -> Optional[models.Prediction]:
        """Create a new prediction"""
        user = self.db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise ValueError("User not found")
        model = self.model_service.get_model_by_id(model_id)
        if not model:
            raise ValueError("Model not found")
        if model.status != models.ModelStatus.TRAINED:
            raise ValueError("Model is not trained yet")
        try:
            start_time = time.time()
            trained_model = self.model_service.load_trained_model(model_id)
            if not trained_model:
                raise ValueError("Failed to load trained model")
            prediction_result = trained_model.predict([input_data])
            try:
                probabilities = trained_model.predict_proba([input_data])
                confidence_score = float(np.max(probabilities))
            except Exception:
                confidence_score = 0.8
            execution_time = int((time.time() - start_time) * 1000)
            if isinstance(prediction_result, np.ndarray):
                prediction_result = prediction_result.tolist()
            prediction = models.Prediction(
                user_id=user_id,
                model_id=model_id,
                input_data=input_data,
                prediction_result=prediction_result,
                confidence_score=confidence_score,
                execution_time_ms=execution_time,
            )
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            return prediction
        except Exception as e:
            logger.error(f"Error creating prediction: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def get_prediction_by_id(self, prediction_id: int) -> Optional[models.Prediction]:
        """Get prediction by ID"""
        return (
            self.db.query(models.Prediction)
            .filter(models.Prediction.id == prediction_id)
            .first()
        )

    def get_predictions_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> List[models.Prediction]:
        """Get predictions by user"""
        return (
            self.db.query(models.Prediction)
            .filter(models.Prediction.user_id == user_id)
            .order_by(desc(models.Prediction.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_predictions_by_model(
        self, model_id: int, skip: int = 0, limit: int = 100
    ) -> List[models.Prediction]:
        """Get predictions by model"""
        return (
            self.db.query(models.Prediction)
            .filter(models.Prediction.model_id == model_id)
            .order_by(desc(models.Prediction.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_all_predictions(
        self, skip: int = 0, limit: int = 100
    ) -> List[models.Prediction]:
        """Get all predictions (admin only)"""
        return (
            self.db.query(models.Prediction)
            .order_by(desc(models.Prediction.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_prediction_statistics(
        self, user_id: int = None, model_id: int = None
    ) -> Dict[str, Any]:
        """Get prediction statistics"""
        query = self.db.query(models.Prediction)
        if user_id:
            query = query.filter(models.Prediction.user_id == user_id)
        if model_id:
            query = query.filter(models.Prediction.model_id == model_id)
        predictions = query.all()
        if not predictions:
            return {
                "total_predictions": 0,
                "avg_confidence": 0,
                "avg_execution_time_ms": 0,
                "predictions_by_model": {},
                "predictions_by_day": {},
            }
        total_predictions = len(predictions)
        avg_confidence = np.mean(
            [p.confidence_score for p in predictions if p.confidence_score]
        )
        avg_execution_time = np.mean(
            [p.execution_time_ms for p in predictions if p.execution_time_ms]
        )
        predictions_by_model: Dict[int, int] = {}
        for prediction in predictions:
            mid = prediction.model_id
            predictions_by_model[mid] = predictions_by_model.get(mid, 0) + 1
        predictions_by_day: Dict[str, int] = {}
        for prediction in predictions:
            day = prediction.created_at.date().isoformat()
            predictions_by_day[day] = predictions_by_day.get(day, 0) + 1
        return {
            "total_predictions": total_predictions,
            "avg_confidence": float(avg_confidence),
            "avg_execution_time_ms": float(avg_execution_time),
            "predictions_by_model": predictions_by_model,
            "predictions_by_day": predictions_by_day,
        }

    def batch_predict(
        self, user_id: int, model_id: int, input_data_list: List[List[float]]
    ) -> List[models.Prediction]:
        """Create multiple predictions in batch"""
        predictions = []
        for input_data in input_data_list:
            try:
                prediction = self.create_prediction(user_id, model_id, input_data)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                continue
        return predictions
