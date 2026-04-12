"""
Hyperparameter optimization for Quantis models using Optuna.

Fixes vs original:
- direction is now validated before study creation (avoid cryptic Optuna error)
- Added MedianPruner support to kill unpromising trials early
- Added best_trial metadata to return value
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = {"minimize", "maximize"}


def create_objective(train_fn: Callable, input_size: int = 128) -> Callable:
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
    train_fn: Callable,
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
        Dict with best_params and best_value, or None if optuna is unavailable.
    """
    # BUG FIX: validate direction before handing to Optuna (clearer error message)
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid direction {direction!r}. Must be one of {VALID_DIRECTIONS}."
        )

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning(
            "optuna not installed; hyperparameter optimisation is unavailable."
        )
        return None

    objective = create_objective(train_fn, input_size)

    # Median pruner: stop trials that are clearly worse than the median at intermediate steps
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
