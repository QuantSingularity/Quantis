import logging
import os
from typing import Any, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import mlflow

    from .mlflow_tracking import log_experiment, register_model
except ImportError:
    log_experiment = None
    register_model = None
    mlflow = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer model for time series forecasting.
    """

    def __init__(
        self, input_size: int = 128, hidden_size: int = 64, output_size: int = 1
    ) -> None:
        super(TemporalFusionTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x: Any) -> Any:
        # Validate input dimensions
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch_size, input_size), got {x.dim()}D tensor"
            )
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {x.shape[1]}")
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class TimeSeriesModel(nn.Module):

    def __init__(
        self, input_size: Any, hidden_size: Any, output_size: Any, num_layers: Any = 2
    ) -> None:
        super(TimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Any) -> Any:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(
    model_or_data_path: Any,
    X_train: Optional[Any] = None,
    y_train: Optional[Any] = None,
    mlflow_tracker: Optional[Any] = None,
    params: Any = None,
) -> Any:
    """
    Train a model. Supports two call signatures:
      1. train_model(model, X_train, y_train, mlflow_tracker)  — for tests
      2. train_model(data_path, params)                         — legacy usage
    """
    # Signature 1: train_model(model, X_train, y_train, mlflow_tracker)
    if isinstance(model_or_data_path, nn.Module):
        model = model_or_data_path
        criterion = nn.MSELoss()
        lr = getattr(model, "learning_rate", 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        num_epochs = getattr(model, "num_epochs", 10)
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        if mlflow_tracker is not None:
            model.eval()
            with torch.no_grad():
                final_outputs = model(X_train)
                final_loss = criterion(final_outputs, y_train).item()
            mlflow_tracker.log_metric("loss", final_loss)

        return model

    # Signature 2: train_model(data_path, params=...)
    if X_train is not None and not isinstance(X_train, dict):
        # Called as train_model(data_path, params_dict)
        params = X_train

    if params is None:
        params = {
            "input_size": 10,
            "hidden_size": 64,
            "output_size": 3,
            "num_layers": 2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        }

    X = np.random.randn(1000, 24, params["input_size"])
    y = np.random.randn(1000, params["output_size"])
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    model = TimeSeriesModel(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        output_size=params["output_size"],
        num_layers=params["num_layers"],
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    for epoch in range(params["epochs"]):
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{params['epochs']}], Loss: {loss.item():.4f}"
            )
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te)
        test_loss = criterion(y_pred, y_te)
        y_test_np = y_te.numpy()
        y_pred_np = y_pred.numpy()
        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        r2 = r2_score(y_test_np, y_pred_np)
    logger.info(f"Test Loss: {test_loss.item():.4f}")
    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    metrics = {"mse": mse, "mae": mae, "r2": r2}

    class ModelWrapper:

        def __init__(self, m):
            self.model = m

        def predict(self, X):
            X_tensor = torch.FloatTensor(X)
            self.model.eval()
            with torch.no_grad():
                return self.model(X_tensor).numpy()

        def predict_proba(self, X):
            preds = self.predict(X)
            total = np.sum(np.abs(preds), axis=1, keepdims=True)
            total = np.where(total == 0, 1, total)
            return np.abs(preds) / total

    wrapped_model = ModelWrapper(model)
    model_path = os.path.join(os.path.dirname(__file__), "tft_model.pkl")
    joblib.dump(wrapped_model, model_path)

    if log_experiment is not None and mlflow is not None:
        try:
            log_experiment(params, metrics, model)
            active_run = mlflow.active_run()
            if active_run and register_model:
                register_model("time_series_forecaster", active_run.info.run_id)
        except Exception as e:
            logger.info(f"MLflow logging failed: {e}")

    return wrapped_model


if __name__ == "__main__":
    train_model(None)
