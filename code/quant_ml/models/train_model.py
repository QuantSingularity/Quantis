"""
Model training utilities for the Quantis ML pipeline.

Fixes applied vs original ml/models/train_model.py:
- Added .detach() before .numpy() calls on gradient-tracked tensors (RuntimeError fix)
- Replaced bare `Any` type hints on nn.Module __init__ params with proper int types
- Fixed train_model(None) crash: guard against None model_or_data_path
- Added gradient clipping to prevent exploding gradients in LSTM
- Rewrote TemporalFusionTransformer with proper GRN blocks, multi-head attention
- Added LR scheduler for adaptive learning
"""

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


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network block — core building block of TFT."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.skip = (
            nn.Linear(input_size, output_size)
            if input_size != output_size
            else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = self.sigmoid(self.gate(h))
        return self.layer_norm(gate * out + (1 - gate) * residual)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer model for time series forecasting.
    Includes GRN blocks, multi-head attention, and gated skip connections.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 64,
        output_size: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10

        self.var_selection = GatedResidualNetwork(
            input_size, hidden_size, hidden_size, dropout
        )
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.positionwise_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch_size, input_size), got {x.dim()}D tensor"
            )
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {x.shape[1]}")
        h = self.var_selection(x)
        h = h.unsqueeze(1)
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + self.dropout(attn_out))
        h = self.positionwise_grn(h.squeeze(1))
        return self.fc_out(h)


class TimeSeriesModel(nn.Module):

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2
    ) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


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
      3. train_model(None)                                      — uses default params
    """
    # BUG FIX: None was passed directly and caused attribute errors downstream
    if model_or_data_path is None:
        model_or_data_path = "default"

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
            # BUG FIX: gradient clipping prevents NaN loss / exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if mlflow_tracker is not None:
            model.eval()
            with torch.no_grad():
                final_outputs = model(X_train)
                final_loss = criterion(final_outputs, y_train).item()
            mlflow_tracker.log_metric("loss", final_loss)
        return model

    # Signature 2: train_model(data_path, params_dict)
    if X_train is not None and not isinstance(X_train, dict):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    for epoch in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{params['epochs']}], Loss: {loss.item():.4f}"
            )
            scheduler.step(loss)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te)
        test_loss = criterion(y_pred, y_te)
        # BUG FIX: .detach() is required before .numpy() on any gradient-tracked tensor
        y_test_np = y_te.detach().numpy()
        y_pred_np = y_pred.detach().numpy()
        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        r2 = r2_score(y_test_np, y_pred_np)

    logger.info(f"Test Loss: {test_loss.item():.4f}")
    logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R\u00b2: {r2:.4f}")
    metrics = {"mse": mse, "mae": mae, "r2": r2}

    class ModelWrapper:
        def __init__(self, m: nn.Module):
            self.model = m

        def predict(self, X: Any) -> np.ndarray:
            X_tensor = torch.FloatTensor(X)
            self.model.eval()
            with torch.no_grad():
                return self.model(X_tensor).detach().numpy()

        def predict_proba(self, X: Any) -> np.ndarray:
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
