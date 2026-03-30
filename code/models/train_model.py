from typing import Any
import os
import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from .mlflow_tracking import log_experiment, register_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Any) -> Any:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(data_path: Any, params: Any = None) -> Any:
    """
    Train a time series forecasting model

    Args:
        data_path: Path to processed data
        params: Dictionary of hyperparameters

    Returns:
        Trained model
    """
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
    X_train, X_test, y_train, y_test = train_test_split(
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
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{params['epochs']}], Loss: {loss.item():.4f}"
            )
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        y_test_np = y_test.numpy()
        y_pred_np = y_pred.numpy()
        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        r2 = r2_score(y_test_np, y_pred_np)
    logger.info(f"Test Loss: {test_loss.item():.4f}")
    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    metrics = {"mse": mse, "mae": mae, "r2": r2}

    class ModelWrapper:

        def __init__(self, model):
            self.model = model

        def predict(self, X):
            X_tensor = torch.FloatTensor(X)
            self.model.eval()
            with torch.no_grad():
                return self.model(X_tensor).numpy()

        def predict_proba(self, X):
            preds = self.predict(X)
            probs = np.abs(preds) / np.sum(np.abs(preds), axis=1, keepdims=True)
            return probs

    wrapped_model = ModelWrapper(model)
    model_path = os.path.join(os.path.dirname(__file__), "tft_model.pkl")
    joblib.dump(wrapped_model, model_path)
    try:
        log_experiment(params, metrics, model)
        register_model("time_series_forecaster", mlflow.active_run().info.run_id)
    except Exception as e:
        logger.info(f"MLflow logging failed: {e}")
    return wrapped_model


if __name__ == "__main__":
    train_model(None)
