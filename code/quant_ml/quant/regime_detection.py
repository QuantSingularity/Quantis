"""
Market Regime Detection
========================
Statistical methods for identifying market regimes (bull/bear/sideways,
high/low volatility, crisis vs. normal) from return time series.

Methods:
- Hidden Markov Model (via hmmlearn)
- Volatility Regime via GARCH(1,1) (via arch)
- Rolling Z-score regime classifier (dependency-free fallback)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    Hidden Markov Model regime detection.

    Fits a Gaussian HMM on log returns and assigns each date to a
    latent market regime (e.g. bull / bear / high-vol).

    Requires: hmmlearn
    """

    def __init__(
        self, n_regimes: int = 2, n_iter: int = 200, random_state: int = 42
    ) -> None:
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None

    def fit(self, returns: pd.Series) -> "HMMRegimeDetector":
        """Fit HMM on a return series."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. "
                "Install with: pip install hmmlearn"
            )
        X = returns.values.reshape(-1, 1)
        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(X)
        return self

    def predict(self, returns: pd.Series) -> pd.Series:
        """Assign regime labels to a return series."""
        if self._model is None:
            raise RuntimeError("Call .fit() before .predict()")
        X = returns.values.reshape(-1, 1)
        states = self._model.predict(X)
        return pd.Series(states, index=returns.index, name="regime")

    def regime_stats(self, returns: pd.Series) -> pd.DataFrame:
        """Summary statistics per regime."""
        regimes = self.predict(returns)
        df = pd.DataFrame({"return": returns, "regime": regimes})
        stats = df.groupby("regime")["return"].agg(
            mean="mean",
            std="std",
            count="count",
            skew=lambda x: x.skew(),
            min="min",
            max="max",
        )
        stats["annualised_return"] = stats["mean"] * 252
        stats["annualised_vol"] = stats["std"] * np.sqrt(252)
        return stats


class GARCHRegimeDetector:
    """
    GARCH(1,1)-based volatility regime classification.

    Fits GARCH on returns, extracts conditional volatility, then
    classifies each period as low / mid / high volatility using
    user-specified percentile thresholds.

    Requires: arch
    """

    def __init__(
        self,
        low_threshold: float = 33.0,
        high_threshold: float = 67.0,
    ) -> None:
        """
        Args:
            low_threshold: Percentile below which vol is "low".
            high_threshold: Percentile above which vol is "high".
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._result = None

    def fit_predict(self, returns: pd.Series) -> pd.DataFrame:
        """
        Fit GARCH(1,1) and return conditional vol + regime label.

        Returns:
            DataFrame with columns: conditional_vol, regime (0=low, 1=mid, 2=high).
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "arch is required for GARCH regime detection. "
                "Install with: pip install arch"
            )
        # Scale returns to percentage for numerical stability
        scaled = returns * 100
        model = arch_model(scaled, vol="Garch", p=1, q=1, dist="Normal")
        self._result = model.fit(disp="off")
        cond_vol = self._result.conditional_volatility / 100  # back to decimal

        low_cut = np.percentile(cond_vol, self.low_threshold)
        high_cut = np.percentile(cond_vol, self.high_threshold)

        regime = pd.Series(1, index=returns.index, name="regime")  # mid
        regime[cond_vol < low_cut] = 0  # low vol
        regime[cond_vol > high_cut] = 2  # high vol

        return pd.DataFrame({"conditional_vol": cond_vol, "regime": regime})


class RollingZScoreRegimeDetector:
    """
    Simple, dependency-free regime detector using rolling Z-score of returns.

    Suitable as a fallback when hmmlearn/arch are unavailable.
    """

    def __init__(self, window: int = 63, threshold: float = 1.0) -> None:
        """
        Args:
            window: Rolling window size (trading days).
            threshold: Z-score threshold for classifying high/low return regimes.
        """
        self.window = window
        self.threshold = threshold

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """
        Classify each period as bull (1), bear (-1), or sideways (0).

        Returns:
            Series of {-1, 0, 1} regime labels.
        """
        rolling_mu = returns.rolling(self.window).mean()
        rolling_std = returns.rolling(self.window).std()
        z = (returns - rolling_mu) / rolling_std.replace(0, np.nan)

        regime = pd.Series(0, index=returns.index, name="regime")
        regime[z > self.threshold] = 1  # bull / momentum
        regime[z < -self.threshold] = -1  # bear / risk-off
        return regime


class RegimeAwarePortfolio:
    """
    Portfolio allocation that shifts weights based on detected market regime.

    Allows rule-based regime-conditional allocation:
    - Different weight vectors per regime
    - Smooth transitions using exponential blending
    """

    def __init__(
        self,
        regime_weights: dict[int, np.ndarray],
        blend_halflife: int = 5,
    ) -> None:
        """
        Args:
            regime_weights: Dict mapping regime label (int) → target weight array.
            blend_halflife: Half-life (days) for EWM blending between regimes.
        """
        self.regime_weights = regime_weights
        self.blend_halflife = blend_halflife

    def compute_weights(self, regimes: pd.Series, n_assets: int) -> pd.DataFrame:
        """
        Compute time-varying portfolio weights given a regime series.

        Args:
            regimes: Series of integer regime labels.
            n_assets: Number of portfolio assets.

        Returns:
            DataFrame of weights (rows=dates, cols=assets).
        """
        raw = np.stack(
            [self.regime_weights.get(r, np.ones(n_assets) / n_assets) for r in regimes]
        )
        weight_df = pd.DataFrame(raw, index=regimes.index)
        # EWM smoothing for smoother transitions
        smoothed = weight_df.ewm(halflife=self.blend_halflife).mean()
        row_sum = smoothed.sum(axis=1).replace(0, 1)
        return smoothed.div(row_sum, axis=0)
