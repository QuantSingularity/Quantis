"""
Backtesting Engine
==================
Event-driven backtesting framework for systematic trading strategies.

Features:
- Vectorised and event-based modes
- Transaction cost modelling (fixed + proportional)
- Slippage model
- Walk-forward / expanding-window validation
- Full performance report via risk_metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .risk_metrics import risk_report

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: float = 1_000_000.0
    commission_bps: float = 10.0  # basis points per trade (one-way)
    slippage_bps: float = 5.0  # basis points of slippage per trade
    rebalance_freq: str = "D"  # pandas offset alias: D, W, M
    max_position_size: float = 0.20  # max weight per asset
    allow_short: bool = False
    risk_free_rate: float = 0.0
    annualisation: int = 252


@dataclass
class BacktestResult:
    """Container for backtest output."""

    portfolio_returns: pd.Series
    portfolio_value: pd.Series
    positions: pd.DataFrame  # weights over time
    turnover: pd.Series  # daily turnover
    transaction_costs: pd.Series
    performance: dict = field(default_factory=dict)


class VectorisedBacktester:
    """
    Fast vectorised backtester.

    Assumes a signal DataFrame that maps to target weights,
    rebalanced at a given frequency with costs and slippage applied.
    """

    def __init__(self, config: BacktestConfig = None) -> None:
        self.cfg = config or BacktestConfig()

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        signal_to_weights: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Run a vectorised backtest.

        Args:
            prices: OHLCV or close-price DataFrame (rows=dates, cols=assets).
            signals: Raw signal DataFrame aligned to prices.
            signal_to_weights: Callable(signals) -> weights DataFrame.
                               Defaults to cross-sectional rank normalisation.

        Returns:
            BacktestResult with full performance attribution.
        """
        if signal_to_weights is None:
            signal_to_weights = self._rank_weights

        raw_weights = signal_to_weights(signals)
        # Clip to max position size and re-normalise
        weights = raw_weights.clip(
            lower=0 if not self.cfg.allow_short else -self.cfg.max_position_size,
            upper=self.cfg.max_position_size,
        )
        row_sum = weights.abs().sum(axis=1).replace(0, 1)
        weights = weights.div(row_sum, axis=0)

        # Resample to rebalance frequency (only possible with DatetimeIndex)
        if isinstance(weights.index, pd.DatetimeIndex):
            weights = (
                weights.resample(self.cfg.rebalance_freq)
                .last()
                .reindex(prices.index, method="ffill")
            )
        else:
            # Integer index fallback: sub-sample every N rows
            _freq_map = {"D": 1, "W": 5, "M": 21, "Q": 63}
            step = _freq_map.get(self.cfg.rebalance_freq, 1)
            mask = pd.Series(False, index=weights.index)
            mask.iloc[::step] = True
            weights = weights.where(mask).ffill()

        # Asset returns
        asset_returns = prices.pct_change()

        # Turnover = sum of abs weight changes
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1)

        # Transaction costs and slippage
        cost_rate = (self.cfg.commission_bps + self.cfg.slippage_bps) / 10_000
        transaction_costs = turnover * cost_rate

        # Portfolio gross returns
        port_returns_gross = (weights.shift(1) * asset_returns).sum(axis=1)
        port_returns_net = port_returns_gross - transaction_costs

        # Portfolio value
        port_value = self.cfg.initial_capital * (1 + port_returns_net).cumprod()

        perf = risk_report(
            port_returns_net.dropna().values,
            risk_free_rate=self.cfg.risk_free_rate,
            annualisation=self.cfg.annualisation,
        )
        perf["total_transaction_costs"] = float(transaction_costs.sum())
        perf["avg_daily_turnover"] = float(turnover.mean())

        return BacktestResult(
            portfolio_returns=port_returns_net,
            portfolio_value=port_value,
            positions=weights,
            turnover=turnover,
            transaction_costs=transaction_costs,
            performance=perf,
        )

    @staticmethod
    def _rank_weights(signals: pd.DataFrame) -> pd.DataFrame:
        """Default: cross-sectional rank normalised to long-only weights."""
        ranks = signals.rank(axis=1, pct=True)
        # Go long top quartile only
        weights = ranks.where(ranks > 0.75, 0)
        row_sum = weights.sum(axis=1).replace(0, 1)
        return weights.div(row_sum, axis=0)


class WalkForwardValidator:
    """
    Walk-forward (expanding window) validation for systematic strategies.

    Splits the dataset into successive in-sample / out-of-sample windows
    and evaluates strategy performance on each out-of-sample period.
    """

    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
        config: BacktestConfig = None,
    ) -> None:
        """
        Args:
            train_size: In-sample training period in rows.
            test_size: Out-of-sample test period in rows.
            step_size: Step between successive windows in rows.
            config: BacktestConfig for the backtester.
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.backtester = VectorisedBacktester(config or BacktestConfig())

    def run(
        self,
        prices: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Execute walk-forward validation.

        Args:
            prices: Full price history.
            signal_fn: Callable(in_sample_prices) -> signals for full dataset.

        Returns:
            DataFrame of out-of-sample performance metrics per fold.
        """
        results = []
        n = len(prices)
        start = self.train_size
        while start + self.test_size <= n:
            train_prices = prices.iloc[:start]
            test_prices = prices.iloc[start : start + self.test_size]

            try:
                signals = signal_fn(train_prices).reindex(
                    test_prices.index, method="ffill"
                )
                bt_result = self.backtester.run(test_prices, signals)
                perf = bt_result.performance.copy()
                # Support both DatetimeIndex and RangeIndex
                idx0 = test_prices.index[0]
                idx1 = test_prices.index[-1]
                perf["fold_start"] = (
                    str(idx0.date()) if hasattr(idx0, "date") else str(idx0)
                )
                perf["fold_end"] = (
                    str(idx1.date()) if hasattr(idx1, "date") else str(idx1)
                )
                results.append(perf)
            except Exception as e:
                logger.warning("Walk-forward fold failed at %s: %s", start, e)

            start += self.step_size

        return pd.DataFrame(results)
