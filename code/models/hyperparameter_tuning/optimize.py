from typing import Any
import optuna
from pytorch_forecasting.models import TemporalFusionTransformer
from ..train_model import train_and_validate


def objective(trial: Any) -> Any:
    params = {
        "learning_rate": trial.suggest_loguniform("lr", 1e-05, 0.01),
        "hidden_size": trial.suggest_categorical("hidden", [256, 512, 1024]),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
    }
    model = TemporalFusionTransformer(**params)
    return train_and_validate(model)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=3600)
