"""
Hyperparameter optimization for Quantis models using Optuna.
"""

from typing import Any, Dict, Optional


def create_objective(train_fn, input_size: int = 128):
    """
    Create an Optuna objective function for the given training function.

    Args:
        train_fn: Callable that takes a params dict and returns a validation score.
        input_size: Model input feature size.

    Returns:
        Objective function suitable for optuna.study.optimize().
    """

    def objective(trial: Any) -> float:
        params = {
            "learning_rate": trial.suggest_float("lr", 1e-5, 0.01, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "input_size": input_size,
        }
        return train_fn(params)

    return objective


def run_optimization(
    train_fn,
    input_size: int = 128,
    n_trials: int = 20,
    timeout: int = 600,
    direction: str = "minimize",
) -> Optional[Dict[str, Any]]:
    """
    Run hyperparameter optimisation.

    Args:
        train_fn: Callable(params) -> float score.
        input_size: Input feature dimension.
        n_trials: Number of Optuna trials.
        timeout: Maximum wall-clock seconds.
        direction: "minimize" or "maximize".

    Returns:
        Best params dict, or None if optuna is not available.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "optuna not installed; hyperparameter optimisation is unavailable."
        )
        return None

    objective = create_objective(train_fn, input_size)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params
