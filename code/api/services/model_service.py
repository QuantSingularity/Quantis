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
            tags=tags or [],
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    # Alias so endpoints calling create_model() work
    def create_model(
        self,
        name: str,
        description: str,
        model_type: str,
        owner_id: int,
        dataset_id: int,
        hyperparameters: Dict = None,
        tags: Optional[List[str]] = None,
    ) -> models.Model:
        return self.create_model_record(
            name=name,
            description=description,
            model_type=model_type,
            owner_id=owner_id,
            dataset_id=dataset_id,
            hyperparameters=hyperparameters,
            tags=tags,
        )

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
            ]:
                setattr(model, key, value)
        model.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(model)
        return model

    # Alias so endpoints calling update_model() work
    def update_model(self, model_id: int, **kwargs) -> Optional[models.Model]:
        return self.update_model_record(model_id, **kwargs)

    def soft_delete_model(self, model_id: int, deleted_by_id: int = None) -> bool:
        model = self.get_model_by_id(model_id)
        if not model:
            return False
        model.is_deleted = True
        model.deleted_at = datetime.utcnow()
        if deleted_by_id is not None:
            model.deleted_by_id = deleted_by_id
        self.db.commit()
        return True

    # Alias so endpoints calling delete_model() work
    def delete_model(self, model_id: int, deleted_by_id: int = None) -> bool:
        return self.soft_delete_model(model_id, deleted_by_id)

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
            X = data.select_dtypes(include=np.number).fillna(0)
            if X.empty:
                raise ValueError("No numeric data found for training.")
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
            if X.empty or y.empty:
                raise ValueError("Insufficient data for training.")
            trained_model = None
            metrics = {}
            model_type = (
                model.model_type.lower()
                if isinstance(model.model_type, str)
                else model.model_type.value.lower()
            )
            dummy_models = {
                "tft": DummyTFTModel,
                "lstm": DummyLSTMModel,
                "arima": DummyARIMAModel,
                "linear_regression": DummyLinearModel,
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

    def train_dummy_model(self, model_id: int) -> bool:
        """Train a dummy model for demo/testing purposes."""
        model = self.get_model_by_id(model_id)
        if not model:
            return False
        try:
            model.status = models.ModelStatus.TRAINING
            self.db.commit()
            dummy = DummyTFTModel()
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            metrics = dummy.train(X, y, model.hyperparameters or {})
            return self.save_trained_model(model_id, dummy, metrics)
        except Exception as e:
            logger.error(f"Error in dummy training for model {model_id}: {e}")
            model.status = models.ModelStatus.FAILED
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
        time.sleep(0.1)
        return {"mse": 0.1, "mae": 0.05, "r2": 0.95}

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return np.random.randn(len(X))
        return np.random.randn(len(X) if hasattr(X, "__len__") else 1)

    def predict_proba(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.dirichlet(np.ones(3), size=n)


class DummyLSTMModel:

    def __init__(self) -> None:
        self.model_type = "LSTM"

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info("Training Dummy LSTM Model")
        return {"mse": 0.12, "mae": 0.06, "r2": 0.93}

    def predict(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.randn(n)

    def predict_proba(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.dirichlet(np.ones(3), size=n)


class DummyARIMAModel:

    def __init__(self) -> None:
        self.model_type = "ARIMA"

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info("Training Dummy ARIMA Model")
        return {"mse": 0.15, "mae": 0.08, "r2": 0.90}

    def predict(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.randn(n)

    def predict_proba(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.dirichlet(np.ones(3), size=n)


class DummyLinearModel:

    def __init__(self) -> None:
        self.model_type = "LinearRegression"
        self.coef_ = None

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info("Training Dummy Linear Model")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if len(X) > 0 and X.shape[1] > 0:
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            self.coef_ = np.zeros(1)
        return {"mse": 0.08, "mae": 0.04, "r2": 0.97}

    def predict(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.coef_ is not None and X.shape[1] == len(self.coef_):
            return X @ self.coef_
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.randn(n)

    def predict_proba(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.dirichlet(np.ones(3), size=n)


class DummyRandomForestModel:

    def __init__(self) -> None:
        self.model_type = "RandomForest"

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info("Training Dummy RandomForest Model")
        return {"mse": 0.09, "mae": 0.045, "r2": 0.96}

    def predict(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.randn(n)

    def predict_proba(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.dirichlet(np.ones(3), size=n)


class DummyXGBoostModel:

    def __init__(self) -> None:
        self.model_type = "XGBoost"

    def train(self, X: Any, y: Any, hyperparameters: Dict = None) -> Any:
        logger.info("Training Dummy XGBoost Model")
        return {"mse": 0.07, "mae": 0.035, "r2": 0.98}

    def predict(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.randn(n)

    def predict_proba(self, X: Any) -> Any:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.random.dirichlet(np.ones(3), size=n)
