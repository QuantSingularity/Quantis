"""
Model service for machine learning model management
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from .. import models
from ..config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


class ModelService:

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_model_record(
        self,
        name: str,
        description: str,
        model_type: str,
        owner_id: int,
        dataset_id: int,
        hyperparameters: Dict = None,
        tags: Optional[List[str]] = None,
    ) -> models.Model:
        """Create a new model record in the database."""
        owner = self.db.query(models.User).filter(models.User.id == owner_id).first()
        if not owner:
            raise ValueError("Owner not found")
        dataset = (
            self.db.query(models.Dataset)
            .filter(
                and_(
                    models.Dataset.id == dataset_id, models.Dataset.is_deleted == False
                )
            )
            .first()
        )
        if not dataset:
            raise ValueError("Dataset not found or is deleted")
        model = models.Model(
            name=name,
            description=description,
            model_type=model_type,
            owner_id=owner_id,
            dataset_id=dataset_id,
            hyperparameters=hyperparameters or {},
            status=models.ModelStatus.CREATED,
            tags=tags or [],
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def get_model_by_id(self, model_id: int) -> Optional[models.Model]:
        return (
            self.db.query(models.Model)
            .filter(and_(models.Model.id == model_id, models.Model.is_deleted == False))
            .first()
        )

    def get_models_by_owner(
        self, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[models.Model]:
        return (
            self.db.query(models.Model)
            .filter(
                and_(
                    models.Model.owner_id == owner_id, models.Model.is_deleted == False
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_all_models(self, skip: int = 0, limit: int = 100) -> List[models.Model]:
        return (
            self.db.query(models.Model)
            .filter(models.Model.is_deleted == False)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_model_record(self, model_id: int, **kwargs) -> Optional[models.Model]:
        model = self.get_model_by_id(model_id)
        if not model:
            return None
        for key, value in kwargs.items():
            if hasattr(model, key) and key not in [
                "id",
                "owner_id",
                "created_at",
                "dataset_id",
                "file_path",
                "metrics",
                "status",
                "trained_at",
            ]:
                setattr(model, key, value)
        self.db.commit()
        self.db.refresh(model)
        return model

    def soft_delete_model(self, model_id: int, deleted_by_id: int) -> bool:
        model = self.get_model_by_id(model_id)
        if not model:
            return False
        model.is_deleted = True
        model.deleted_at = datetime.utcnow()
        model.deleted_by_id = deleted_by_id
        self.db.commit()
        return True

    def save_trained_model(
        self, model_id: int, trained_model: Any, metrics: Dict = None
    ) -> bool:
        model = self.get_model_by_id(model_id)
        if not model:
            return False
        try:
            storage_dir = settings.storage_directory
            os.makedirs(storage_dir, exist_ok=True)
            file_path = os.path.join(storage_dir, f"model_{model_id}.pkl")
            joblib.dump(trained_model, file_path)
            model.file_path = file_path
            model.status = models.ModelStatus.TRAINED
            model.trained_at = datetime.utcnow()
            if metrics:
                model.metrics = metrics
            self.db.commit()
            logger.info(f"Model {model_id} saved and status updated to TRAINED.")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
            model.status = models.ModelStatus.FAILED
            self.db.commit()
            return False

    def load_trained_model(self, model_id: int) -> Optional[Any]:
        model = self.get_model_by_id(model_id)
        if (
            not model
            or not model.file_path
            or model.status != models.ModelStatus.TRAINED
        ):
            logger.warning(
                f"Model {model_id} not found, not trained, or file path missing."
            )
            return None
        try:
            return joblib.load(model.file_path)
        except Exception as e:
            logger.error(f"Error loading model {model_id} from {model.file_path}: {e}")
            return None

    def train_model(self, model_id: int, data: pd.DataFrame) -> bool:
        model = self.get_model_by_id(model_id)
        if not model:
            logger.error(f"Model {model_id} not found for training.")
            return False
        try:
            model.status = models.ModelStatus.TRAINING
            self.db.commit()
            logger.info(f"Model {model_id} status updated to TRAINING.")
            time.sleep(np.random.uniform(5, 15))
            X = data.select_dtypes(include=np.number).fillna(0)
            if X.empty:
                raise ValueError("No numeric data found for training.")
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
            if X.empty or y.empty:
                raise ValueError("Insufficient data for training.")
            trained_model = None
            metrics = {}
            model_type = model.model_type.lower()
            dummy_models = {
                "tft": DummyTFTModel,
                "lstm": DummyLSTMModel,
                "arima": DummyARIMAModel,
                "linear": DummyLinearModel,
                "random_forest": DummyRandomForestModel,
                "xgboost": DummyXGBoostModel,
            }
            if model_type in dummy_models:
                trained_model = dummy_models[model_type]()
                metrics.update(trained_model.train(X, y, model.hyperparameters))
            else:
                raise ValueError(f"Unsupported model type: {model.model_type}")
            return self.save_trained_model(model_id, trained_model, metrics)
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            model.status = models.ModelStatus.FAILED
            model.metrics = {"error": str(e)}
            self.db.commit()
            return False

    def predict_with_model(
        self, model_id: int, input_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        model = self.get_model_by_id(model_id)
        if not model or model.status != models.ModelStatus.TRAINED:
            logger.error(f"Model {model_id} not found or not trained for prediction.")
            return None
        trained_model = self.load_trained_model(model_id)
        if not trained_model:
            return None
        try:
            predictions = trained_model.predict(input_data)
            return pd.DataFrame(predictions, columns=["prediction"])
        except Exception as e:
            logger.error(f"Error during prediction for model {model_id}: {e}")
            return None


class DummyTFTModel:

    def __init__(self) -> None:
        self.model_type = "TFT"
        self.weights = np.random.randn(10, 5)

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info(f"Training Dummy TFT Model with hyperparameters: {hyperparameters}")
        time.sleep(1)
        return {
            "mse": np.random.uniform(0.1, 0.5),
            "mae": np.random.uniform(0.05, 0.3),
            "rmse": np.random.uniform(0.2, 0.7),
            "r2_score": np.random.uniform(0.7, 0.95),
            "training_time": np.random.uniform(10, 300),
            "epochs": hyperparameters.get("epochs", 100) if hyperparameters else 100,
        }

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if X.shape[1] < 10:
            X = np.pad(X, ((0, 0), (0, 10 - X.shape[1])), mode="constant")
        elif X.shape[1] > 10:
            X = X[:, :10]
        return np.dot(X, self.weights)


class DummyLSTMModel:

    def __init__(self) -> None:
        self.model_type = "LSTM"
        self.weights = np.random.randn(8, 3)

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info(
            f"Training Dummy LSTM Model with hyperparameters: {hyperparameters}"
        )
        time.sleep(1)
        return {
            "mse": np.random.uniform(0.1, 0.5),
            "mae": np.random.uniform(0.05, 0.3),
            "rmse": np.random.uniform(0.2, 0.7),
            "r2_score": np.random.uniform(0.7, 0.95),
            "training_time": np.random.uniform(10, 300),
            "epochs": hyperparameters.get("epochs", 50) if hyperparameters else 50,
        }

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if X.shape[1] < 8:
            X = np.pad(X, ((0, 0), (0, 8 - X.shape[1])), mode="constant")
        elif X.shape[1] > 8:
            X = X[:, :8]
        return np.dot(X, self.weights)


class DummyARIMAModel:

    def __init__(self) -> None:
        self.model_type = "ARIMA"
        self.coefficients = np.random.randn(5)

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info(
            f"Training Dummy ARIMA Model with hyperparameters: {hyperparameters}"
        )
        time.sleep(0.5)
        return {
            "mse": np.random.uniform(0.1, 0.5),
            "mae": np.random.uniform(0.05, 0.3),
            "rmse": np.random.uniform(0.2, 0.7),
            "training_time": np.random.uniform(5, 60),
        }

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        result = np.sum(X * self.coefficients[: X.shape[1]], axis=1, keepdims=True)
        return result.reshape(-1, 1)


class DummyLinearModel:

    def __init__(self) -> None:
        self.model_type = "Linear"
        self.weights = np.random.randn(6)
        self.bias = np.random.randn()

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info(
            f"Training Dummy Linear Model with hyperparameters: {hyperparameters}"
        )
        time.sleep(0.2)
        return {
            "mse": np.random.uniform(0.1, 0.5),
            "mae": np.random.uniform(0.05, 0.3),
            "rmse": np.random.uniform(0.2, 0.7),
            "r2_score": np.random.uniform(0.7, 0.95),
            "training_time": np.random.uniform(1, 30),
        }

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if X.shape[1] < 6:
            X = np.pad(X, ((0, 0), (0, 6 - X.shape[1])), mode="constant")
        elif X.shape[1] > 6:
            X = X[:, :6]
        return (np.dot(X, self.weights) + self.bias).reshape(-1, 1)


class DummyRandomForestModel:

    def __init__(self) -> None:
        self.model_type = "RandomForest"
        self.n_estimators = 100
        self.features = None

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info(
            f"Training Dummy Random Forest Model with hyperparameters: {hyperparameters}"
        )
        self.n_estimators = (
            hyperparameters.get("n_estimators", 100) if hyperparameters else 100
        )
        self.features = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        time.sleep(1.5)
        return {
            "mse": np.random.uniform(0.05, 0.3),
            "mae": np.random.uniform(0.02, 0.15),
            "r2_score": np.random.uniform(0.8, 0.98),
            "training_time": np.random.uniform(20, 400),
        }

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame) and self.features:
            X = X[self.features].values
        return np.random.rand(X.shape[0], 1) * 100


class DummyXGBoostModel:

    def __init__(self) -> None:
        self.model_type = "XGBoost"
        self.n_estimators = 100
        self.features = None

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info(
            f"Training Dummy XGBoost Model with hyperparameters: {hyperparameters}"
        )
        self.n_estimators = (
            hyperparameters.get("n_estimators", 100) if hyperparameters else 100
        )
        self.features = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        time.sleep(2)
        return {
            "mse": np.random.uniform(0.03, 0.2),
            "mae": np.random.uniform(0.01, 0.1),
            "r2_score": np.random.uniform(0.85, 0.99),
            "training_time": np.random.uniform(30, 600),
        }

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame) and self.features:
            X = X[self.features].values
        return np.random.rand(X.shape[0], 1) * 100
