"""
Advanced Quantitative Risk Metrics
===================================
Implements industry-standard risk and performance metrics used in quantitative
finance: VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown, Beta, and more.

All functions accept numpy arrays or pandas Series of *returns* (not prices).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.Series]


def _to_array(x: ArrayLike) -> np.ndarray:
    """Convert input to a clean 1-D float array, dropping NaNs."""
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


# ---------------------------------------------------------------------------
# Value at Risk & Expected Shortfall
# ---------------------------------------------------------------------------


def value_at_risk(
    returns: ArrayLike,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Compute Value at Risk.

    Args:
        returns: Array of period returns (e.g. daily).
        confidence: Confidence level (e.g. 0.95 for 95% VaR).
        method: 'historical' | 'parametric' | 'cornish_fisher'

    Returns:
        VaR as a positive number (loss).
    """
    r = _to_array(returns)
    alpha = 1 - confidence

    if method == "historical":
        return float(-np.percentile(r, 100 * alpha))

    elif method == "parametric":
        mu, sigma = r.mean(), r.std(ddof=1)
        z = stats.norm.ppf(alpha)
        return float(-(mu + z * sigma))

    elif method == "cornish_fisher":
        # Adjusted VaR using skewness and excess kurtosis
        mu, sigma = r.mean(), r.std(ddof=1)
        s = stats.skew(r)
        k = stats.kurtosis(r)  # excess kurtosis
        z = stats.norm.ppf(alpha)
        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )
        return float(-(mu + z_cf * sigma))

    else:
        raise ValueError(
            f"Unknown VaR method: {method!r}. Use 'historical', 'parametric', or 'cornish_fisher'."
        )


def conditional_value_at_risk(
    returns: ArrayLike,
    confidence: float = 0.95,
) -> float:
    """
    Expected Shortfall (CVaR / ES): average loss beyond VaR.

    Args:
        returns: Array of period returns.
        confidence: Confidence level.

    Returns:
        CVaR as a positive number.
    """
    r = _to_array(returns)
    alpha = 1 - confidence
    threshold = np.percentile(r, 100 * alpha)
    tail = r[r <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def max_drawdown(returns: ArrayLike) -> float:
    """
    Maximum drawdown of a return series.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.25 = 25% peak-to-trough).
    """
    r = _to_array(returns)
    cumulative = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(-drawdowns.min())


def drawdown_series(returns: ArrayLike) -> np.ndarray:
    """Return the full drawdown time series (negative values)."""
    r = _to_array(returns)
    cumulative = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumulative)
    return (cumulative - running_max) / running_max


def calmar_ratio(returns: ArrayLike, annualisation: int = 252) -> float:
    """
    Calmar Ratio: annualised return divided by max drawdown.

    Returns:
        Calmar ratio, or np.nan if max drawdown is zero.
    """
    r = _to_array(returns)
    annual_return = np.prod(1 + r) ** (annualisation / len(r)) - 1
    mdd = max_drawdown(r)
    return float(annual_return / mdd) if mdd != 0 else float("nan")


# ---------------------------------------------------------------------------
# Risk-Adjusted Performance
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualisation: int = 252,
) -> float:
    """
    Annualised Sharpe Ratio.

    Args:
        returns: Periodic returns.
        risk_free_rate: Annual risk-free rate (converted to per-period internally).
        annualisation: Trading periods per year (252 for daily equity).

    Returns:
        Annualised Sharpe ratio.
    """
    r = _to_array(returns)
    rf_per_period = (1 + risk_free_rate) ** (1 / annualisation) - 1
    excess = r - rf_per_period
    if excess.std(ddof=1) == 0:
        return float("nan")
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(annualisation))


def sortino_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualisation: int = 252,
    target: float = 0.0,
) -> float:
    """
    Annualised Sortino Ratio (uses downside deviation only).

    Args:
        returns: Periodic returns.
        risk_free_rate: Annual risk-free rate.
        annualisation: Trading periods per year.
        target: Minimum acceptable return per period.

    Returns:
        Annualised Sortino ratio.
    """
    r = _to_array(returns)
    rf_per_period = (1 + risk_free_rate) ** (1 / annualisation) - 1
    excess = r - rf_per_period
    downside = np.minimum(r - target, 0)
    downside_vol = np.sqrt(np.mean(downside**2)) * np.sqrt(annualisation)
    if downside_vol == 0:
        return float("nan")
    return float(excess.mean() * annualisation / downside_vol)


def omega_ratio(
    returns: ArrayLike,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio: probability-weighted gains vs losses relative to threshold.

    Returns:
        Omega ratio (>1 is desirable).
    """
    r = _to_array(returns)
    gains = np.sum(np.maximum(r - threshold, 0))
    losses = np.sum(np.maximum(threshold - r, 0))
    return float(gains / losses) if losses != 0 else float("inf")


# ---------------------------------------------------------------------------
# Market Exposure
# ---------------------------------------------------------------------------


def beta(
    portfolio_returns: ArrayLike,
    benchmark_returns: ArrayLike,
) -> float:
    """
    Portfolio beta relative to a benchmark.

    Returns:
        Beta coefficient.
    """
    p = _to_array(portfolio_returns)
    b = _to_array(benchmark_returns)
    n = min(len(p), len(b))
    p, b = p[:n], b[:n]
    cov_matrix = np.cov(p, b, ddof=1)
    bench_var = cov_matrix[1, 1]
    return float(cov_matrix[0, 1] / bench_var) if bench_var != 0 else float("nan")


def alpha(
    portfolio_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualisation: int = 252,
) -> float:
    """
    Jensen's Alpha: annualised excess return adjusted for market beta.

    Returns:
        Alpha (annualised).
    """
    p = _to_array(portfolio_returns)
    b = _to_array(benchmark_returns)
    n = min(len(p), len(b))
    p, b = p[:n], b[:n]
    rf = (1 + risk_free_rate) ** (1 / annualisation) - 1
    b_val = beta(p, b)
    port_mean = p.mean()
    bench_mean = b.mean()
    raw_alpha = port_mean - rf - b_val * (bench_mean - rf)
    return float(raw_alpha * annualisation)


def information_ratio(
    portfolio_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    annualisation: int = 252,
) -> float:
    """
    Information Ratio: active return over tracking error.

    Returns:
        Annualised information ratio.
    """
    p = _to_array(portfolio_returns)
    b = _to_array(benchmark_returns)
    n = min(len(p), len(b))
    active = p[:n] - b[:n]
    te = active.std(ddof=1)
    if te == 0:
        return float("nan")
    return float(active.mean() / te * np.sqrt(annualisation))


# ---------------------------------------------------------------------------
# Full Risk Report
# ---------------------------------------------------------------------------


def risk_report(
    returns: ArrayLike,
    benchmark_returns: Optional[ArrayLike] = None,
    risk_free_rate: float = 0.0,
    annualisation: int = 252,
    confidence: float = 0.95,
) -> dict:
    """
    Generate a comprehensive risk/performance report dictionary.

    Args:
        returns: Portfolio return series.
        benchmark_returns: Optional benchmark return series.
        risk_free_rate: Annual risk-free rate.
        annualisation: Periods per year.
        confidence: VaR/CVaR confidence level.

    Returns:
        Dictionary of risk metrics.
    """
    r = _to_array(returns)
    report = {
        "total_return": float(np.prod(1 + r) - 1),
        "annualised_return": float(np.prod(1 + r) ** (annualisation / len(r)) - 1),
        "annualised_volatility": float(r.std(ddof=1) * np.sqrt(annualisation)),
        "sharpe_ratio": sharpe_ratio(r, risk_free_rate, annualisation),
        "sortino_ratio": sortino_ratio(r, risk_free_rate, annualisation),
        "calmar_ratio": calmar_ratio(r, annualisation),
        "omega_ratio": omega_ratio(r),
        "max_drawdown": max_drawdown(r),
        "var_historical": value_at_risk(r, confidence, "historical"),
        "var_parametric": value_at_risk(r, confidence, "parametric"),
        "cvar": conditional_value_at_risk(r, confidence),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "win_rate": float(np.mean(r > 0)),
    }
    if benchmark_returns is not None:
        b = _to_array(benchmark_returns)
        report["beta"] = beta(r, b)
        report["alpha"] = alpha(r, b, risk_free_rate, annualisation)
        report["information_ratio"] = information_ratio(r, b, annualisation)
    return report
