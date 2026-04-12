"""
Alpha Signal Generation
========================
Cross-sectional and time-series alpha factors used in quantitative equity
and systematic trading strategies.

All signal functions accept a pandas DataFrame where:
  - rows are time steps (DatetimeIndex preferred)
  - columns are assets/tickers

Returns are normalised (z-scored) within each cross-section unless noted.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score with outlier clipping."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-clip, clip)


# ---------------------------------------------------------------------------
# Momentum Factors
# ---------------------------------------------------------------------------


def momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Cross-sectional price momentum (12-1 month).

    Args:
        prices: Asset price DataFrame.
        lookback: Formation window in trading days.
        skip: Skip most recent `skip` days to avoid short-term reversal.

    Returns:
        Cross-sectionally z-scored momentum scores.
    """
    signal = prices.shift(skip) / prices.shift(lookback) - 1
    return _cs_zscore(signal)


def time_series_momentum(
    returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Time-series momentum: sign of cumulative return over lookback.
    Long if positive cum-return, short if negative (TSMOM).

    Returns:
        {-1, +1} signal per asset per date.
    """
    cum_ret = returns.rolling(lookback).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    return cum_ret.apply(np.sign)


def short_term_reversal(
    returns: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    Short-term reversal: negative of recent return (contrarian signal).

    Returns:
        Cross-sectionally z-scored reversal scores.
    """
    signal = -returns.rolling(lookback).sum()
    return _cs_zscore(signal)


# ---------------------------------------------------------------------------
# Value Factors
# ---------------------------------------------------------------------------


def book_to_market(
    book_value: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classic value factor: book-to-market ratio.

    Returns:
        Cross-sectionally z-scored B/M scores.
    """
    btm = book_value / market_cap.replace(0, np.nan)
    return _cs_zscore(btm)


def earnings_yield(
    earnings: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Earnings-to-price (E/P) value factor.

    Returns:
        Cross-sectionally z-scored E/P scores.
    """
    ep = earnings / market_cap.replace(0, np.nan)
    return _cs_zscore(ep)


# ---------------------------------------------------------------------------
# Quality Factors
# ---------------------------------------------------------------------------


def profitability(
    roa: pd.DataFrame,
    roe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Profitability composite: average of ROA (and ROE if provided).

    Returns:
        Cross-sectionally z-scored profitability scores.
    """
    if roe is not None:
        combo = (roa + roe) / 2
    else:
        combo = roa
    return _cs_zscore(combo)


def low_volatility(
    returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Low-volatility anomaly factor: lower realised vol → higher score.

    Returns:
        Cross-sectionally z-scored low-vol scores (negative of vol).
    """
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    return _cs_zscore(-vol)


# ---------------------------------------------------------------------------
# Statistical Arbitrage / Mean Reversion
# ---------------------------------------------------------------------------


def z_score_mean_reversion(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Z-score of price relative to its rolling mean — classic stat-arb signal.
    Negative z-score → expected reversion upward (buy signal).

    Returns:
        Z-score series per asset.
    """
    mu = prices.rolling(lookback).mean()
    sigma = prices.rolling(lookback).std()
    return -(prices - mu) / sigma.replace(0, np.nan)


def pairs_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Rolling OLS spread between two cointegrated assets.

    Returns:
        DataFrame with columns: spread, hedge_ratio, z_score.
    """
    from sklearn.linear_model import LinearRegression

    out = []
    for i in range(lookback, len(price_a)):
        y = price_a.iloc[i - lookback : i].values.reshape(-1, 1)
        x = price_b.iloc[i - lookback : i].values.reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True).fit(x, y)
        hedge = float(reg.coef_[0][0])
        spread_val = price_a.iloc[i] - hedge * price_b.iloc[i]
        out.append(
            {
                "date": price_a.index[i],
                "spread": spread_val,
                "hedge_ratio": hedge,
            }
        )

    df = pd.DataFrame(out).set_index("date")
    mu = df["spread"].mean()
    sigma = df["spread"].std()
    df["z_score"] = (df["spread"] - mu) / (sigma if sigma != 0 else 1)
    return df


# ---------------------------------------------------------------------------
# Signal Combination
# ---------------------------------------------------------------------------


def combine_signals(
    signals: dict[str, pd.DataFrame],
    weights: Optional[dict[str, float]] = None,
    normalise: bool = True,
) -> pd.DataFrame:
    """
    Combine multiple alpha signals into a composite score.

    Args:
        signals: Dict mapping signal name → signal DataFrame (same shape).
        weights: Optional dict of per-signal weights. Equal-weighted if None.
        normalise: If True, z-score each signal before combining.

    Returns:
        Composite signal DataFrame.
    """
    names = list(signals.keys())
    if weights is None:
        weights = {k: 1.0 / len(names) for k in names}

    composite = None
    for name, sig in signals.items():
        s = _cs_zscore(sig) if normalise else sig
        weighted = s * weights.get(name, 1.0)
        composite = weighted if composite is None else composite + weighted

    return composite
