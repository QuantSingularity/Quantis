"""
Execution & Transaction Cost Model
====================================
Models realistic execution costs for algorithmic trading strategies:
- Market Impact (Almgren-Chriss linear permanent + temporary impact)
- Spread costs
- Optimal trade scheduling (VWAP / TWAP slicing)
- Pre-trade cost estimation

Reference: Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2000).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionParams:
    """Market microstructure parameters for a single asset."""

    symbol: str
    avg_daily_volume: float  # shares / contracts
    bid_ask_spread_bps: float = 10.0  # half-spread in bps
    volatility_daily: float = 0.02  # daily return vol
    price: float = 100.0  # reference price
    eta: float = 0.1  # temporary impact coefficient
    gamma: float = 0.1  # permanent impact coefficient


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model.

    Estimates temporary and permanent price impact for executing
    a block order over a specified horizon, and computes the
    optimal trajectory that minimises expected cost + variance.
    """

    def __init__(self, params: ExecutionParams) -> None:
        self.p = params

    def temporary_impact(self, trade_rate: float) -> float:
        """
        Temporary impact (bps) for a given daily trade rate (fraction of ADV).

        Args:
            trade_rate: Shares traded per day as a fraction of ADV.

        Returns:
            Temporary impact cost in bps.
        """
        return self.p.eta * trade_rate * 10_000

    def permanent_impact(self, total_fraction: float) -> float:
        """
        Permanent impact (bps) for total order as fraction of ADV.

        Args:
            total_fraction: Total order size as fraction of ADV.

        Returns:
            Permanent impact in bps.
        """
        return self.p.gamma * total_fraction * 10_000

    def optimal_trajectory(
        self,
        order_shares: float,
        horizon_days: int = 5,
        risk_aversion: float = 1e-6,
    ) -> pd.DataFrame:
        """
        Compute the optimal execution trajectory (AC model).

        Args:
            order_shares: Total shares to execute (positive=buy, negative=sell).
            horizon_days: Number of days over which to execute.
            risk_aversion: Lambda (risk aversion coefficient) — higher = faster execution.

        Returns:
            DataFrame with columns: day, shares_to_trade, cumulative_executed,
            temp_impact_bps, perm_impact_bps, total_cost_bps.
        """
        T = horizon_days
        sigma = self.p.volatility_daily
        adv = self.p.avg_daily_volume
        eta = self.p.eta / adv
        self.p.gamma / adv

        # Characteristic decay time
        kappa_sq = risk_aversion * sigma**2 / eta
        kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 1e-6

        # Optimal trajectory (discrete time)
        days = np.arange(1, T + 1)
        sinh_kT = np.sinh(kappa * T)
        trajectory = (
            order_shares * np.sinh(kappa * (T - days + 1)) / sinh_kT
            if sinh_kT != 0
            else np.full(T, order_shares / T)
        )
        daily_trade = np.diff(np.concatenate([[order_shares], trajectory]))
        daily_trade = -daily_trade  # shares executed each day

        records = []
        cum_exec = 0.0
        for i, (day, trade) in enumerate(zip(days, daily_trade)):
            cum_exec += trade
            rate = abs(trade) / adv
            ti = self.temporary_impact(rate)
            pi = (
                self.permanent_impact(abs(order_shares) / adv) * (trade / order_shares)
                if order_shares != 0
                else 0
            )
            records.append(
                {
                    "day": int(day),
                    "shares_to_trade": round(trade, 2),
                    "cumulative_executed": round(cum_exec, 2),
                    "temp_impact_bps": round(ti, 4),
                    "perm_impact_bps": round(abs(pi), 4),
                    "total_cost_bps": round(
                        ti + abs(pi) + self.p.bid_ask_spread_bps / 2, 4
                    ),
                }
            )
        return pd.DataFrame(records)

    def pre_trade_estimate(self, order_shares: float, horizon_days: int = 5) -> dict:
        """
        Estimate total execution cost before trading.

        Returns:
            Dict with expected cost metrics.
        """
        traj = self.optimal_trajectory(order_shares, horizon_days)
        total_cost_bps = traj["total_cost_bps"].sum()
        order_value = abs(order_shares) * self.p.price
        cost_dollars = order_value * total_cost_bps / 10_000
        participation_rate = abs(order_shares) / (
            self.p.avg_daily_volume * horizon_days
        )
        return {
            "symbol": self.p.symbol,
            "order_shares": order_shares,
            "order_value_usd": order_value,
            "horizon_days": horizon_days,
            "total_impact_bps": round(total_cost_bps, 4),
            "total_cost_usd": round(cost_dollars, 2),
            "participation_rate": round(participation_rate, 4),
            "spread_cost_bps": self.p.bid_ask_spread_bps / 2,
        }


class TWAPScheduler:
    """Time-Weighted Average Price execution scheduler."""

    @staticmethod
    def schedule(
        order_shares: float,
        horizon_intervals: int,
    ) -> np.ndarray:
        """
        Slice order into equal-sized tranches.

        Args:
            order_shares: Total shares to execute.
            horizon_intervals: Number of execution intervals.

        Returns:
            Array of per-interval trade sizes.
        """
        base = order_shares / horizon_intervals
        schedule = np.full(horizon_intervals, base)
        # Distribute rounding residual to the last interval
        schedule[-1] += order_shares - schedule.sum()
        return schedule


class VWAPScheduler:
    """Volume-Weighted Average Price execution scheduler."""

    @staticmethod
    def schedule(
        order_shares: float,
        volume_profile: np.ndarray,
    ) -> np.ndarray:
        """
        Scale order to match intraday volume profile.

        Args:
            order_shares: Total shares to execute.
            volume_profile: Array of fractional volume per interval (must sum to ~1).

        Returns:
            Array of per-interval trade sizes.
        """
        profile = np.asarray(volume_profile, dtype=float)
        profile = profile / profile.sum()
        schedule = order_shares * profile
        schedule[-1] += order_shares - schedule.sum()  # handle rounding
        return schedule
