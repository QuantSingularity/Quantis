"""
Portfolio Optimisation
======================
Mean-Variance (Markowitz), Risk Parity, and Black-Litterman portfolio
construction — the three pillars of quantitative portfolio management.

Requires: numpy, pandas, scipy.
Optional: cvxpy (for constrained MVO).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """
    Markowitz Mean-Variance Optimisation.

    Supports:
    - Maximum Sharpe Ratio (tangency portfolio)
    - Minimum Variance portfolio
    - Target-return efficient frontier point
    - Long-only and long/short constraints
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        annualisation: int = 252,
    ) -> None:
        """
        Args:
            returns: DataFrame of asset returns (rows=dates, cols=assets).
            risk_free_rate: Annual risk-free rate.
            annualisation: Trading periods per year.
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.annualisation = annualisation
        self.n = returns.shape[1]
        self.mu = returns.mean().values * annualisation  # annualised expected returns
        self.Sigma = (
            returns.cov().values * annualisation
        )  # annualised covariance matrix
        self.asset_names: List[str] = list(returns.columns)

    def _portfolio_stats(self, w: np.ndarray) -> Tuple[float, float, float]:
        """Return (expected_return, volatility, sharpe)."""
        port_return = float(w @ self.mu)
        port_vol = float(np.sqrt(w @ self.Sigma @ w))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        return port_return, port_vol, sharpe

    def max_sharpe(self, allow_short: bool = False) -> Dict:
        """Find the maximum Sharpe Ratio portfolio."""
        w0 = np.ones(self.n) / self.n
        bounds = ((-1, 1) if allow_short else (0, 1),) * self.n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        def neg_sharpe(w):
            r, v, s = self._portfolio_stats(w)
            return -s

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if not result.success:
            logger.warning(
                "max_sharpe optimisation did not converge: %s", result.message
            )
        w_opt = result.x
        r, v, s = self._portfolio_stats(w_opt)
        return {
            "weights": dict(zip(self.asset_names, w_opt)),
            "expected_return": r,
            "volatility": v,
            "sharpe_ratio": s,
            "method": "max_sharpe",
        }

    def min_variance(self, allow_short: bool = False) -> Dict:
        """Find the global minimum variance portfolio."""
        w0 = np.ones(self.n) / self.n
        bounds = ((-1, 1) if allow_short else (0, 1),) * self.n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        def port_var(w):
            return float(w @ self.Sigma @ w)

        result = minimize(
            port_var,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w_opt = result.x
        r, v, s = self._portfolio_stats(w_opt)
        return {
            "weights": dict(zip(self.asset_names, w_opt)),
            "expected_return": r,
            "volatility": v,
            "sharpe_ratio": s,
            "method": "min_variance",
        }

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Trace the efficient frontier.

        Returns:
            DataFrame with columns: [expected_return, volatility, sharpe_ratio, *asset_names]
        """
        min_ret = self.mu.min()
        max_ret = self.mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        records = []
        for target in target_returns:
            w0 = np.ones(self.n) / self.n
            bounds = ((0, 1),) * self.n
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ self.mu - t},
            ]

            def port_var(w):
                return float(w @ self.Sigma @ w)

            res = minimize(
                port_var,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.success:
                w = res.x
                r, v, s = self._portfolio_stats(w)
                row = {"expected_return": r, "volatility": v, "sharpe_ratio": s}
                row.update(dict(zip(self.asset_names, w)))
                records.append(row)
        return pd.DataFrame(records)


class RiskParityOptimizer:
    """
    Risk Parity (Equal Risk Contribution) portfolio.

    Each asset contributes equally to total portfolio variance.
    Standard in multi-asset and macro quant funds.
    """

    def __init__(self, returns: pd.DataFrame, annualisation: int = 252) -> None:
        self.returns = returns
        self.n = returns.shape[1]
        self.Sigma = returns.cov().values * annualisation
        self.asset_names = list(returns.columns)

    def _risk_contributions(self, w: np.ndarray) -> np.ndarray:
        port_var = float(w @ self.Sigma @ w)
        marginal_rc = self.Sigma @ w
        rc = w * marginal_rc / port_var
        return rc

    def optimize(self) -> Dict:
        """Find equal risk contribution weights."""
        w0 = np.ones(self.n) / self.n
        target_rc = np.ones(self.n) / self.n

        def objective(w):
            rc = self._risk_contributions(w)
            return float(np.sum((rc - target_rc) ** 2))

        bounds = ((1e-6, 1.0),) * self.n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w_opt = result.x / result.x.sum()
        rc = self._risk_contributions(w_opt)
        port_vol = float(np.sqrt(w_opt @ self.Sigma @ w_opt))
        return {
            "weights": dict(zip(self.asset_names, w_opt)),
            "risk_contributions": dict(zip(self.asset_names, rc)),
            "volatility": port_vol,
            "method": "risk_parity",
        }


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio construction.

    Blends market equilibrium returns (from market-cap weights) with
    investor views (absolute or relative) to produce posterior return estimates,
    then runs MVO on the posterior.

    Reference: He & Litterman (1999).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_weights: np.ndarray,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        annualisation: int = 252,
    ) -> None:
        """
        Args:
            returns: Historical asset returns DataFrame.
            market_weights: Market-cap weights (must sum to 1).
            risk_aversion: Risk aversion coefficient (lambda).
            tau: Uncertainty scalar on equilibrium returns (typically 0.01–0.10).
            annualisation: Trading periods per year.
        """
        self.returns = returns
        self.n = returns.shape[1]
        self.Sigma = returns.cov().values * annualisation
        self.asset_names = list(returns.columns)
        self.w_mkt = np.asarray(market_weights, dtype=float)
        self.risk_aversion = risk_aversion
        self.tau = tau
        # Implied equilibrium excess returns (reverse optimisation)
        self.pi = risk_aversion * self.Sigma @ self.w_mkt

    def incorporate_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
        view_uncertainty: float = 0.1,
    ) -> Dict:
        """
        Compute Black-Litterman posterior returns and run MVO.

        Args:
            P: Pick matrix (k x n): each row encodes one view across assets.
            Q: View return vector (k,): expected return for each view.
            Omega: View uncertainty matrix (k x k). If None, proportional to P*Sigma*P'.
            view_uncertainty: Scaling factor for auto-Omega (used only when Omega is None).

        Returns:
            Dict with posterior_mu, weights, volatility, sharpe_ratio.
        """
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        tSigma = self.tau * self.Sigma
        if Omega is None:
            Omega = view_uncertainty * P @ tSigma @ P.T

        # BL posterior formulae
        M_inv = np.linalg.inv(np.linalg.inv(tSigma) + P.T @ np.linalg.inv(Omega) @ P)
        posterior_mu = M_inv @ (
            np.linalg.inv(tSigma) @ self.pi + P.T @ np.linalg.inv(Omega) @ Q
        )
        posterior_Sigma = self.Sigma + M_inv

        # MVO on posterior
        mvo = MeanVarianceOptimizer(
            pd.DataFrame(
                np.zeros((1, self.n)), columns=self.asset_names
            ),  # placeholder
        )
        mvo.mu = posterior_mu
        mvo.Sigma = posterior_Sigma
        mvo.n = self.n
        result = mvo.max_sharpe()
        result["posterior_mu"] = dict(zip(self.asset_names, posterior_mu))
        result["method"] = "black_litterman"
        return result
