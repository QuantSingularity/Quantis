"""
quant_ml.quant — Advanced Quantitative Finance Module
======================================================

Submodules
----------
risk_metrics        : VaR, CVaR, Sharpe, Sortino, Calmar, drawdown, alpha, beta, IR
portfolio_optimizer : Mean-Variance (Markowitz), Risk Parity, Black-Litterman
alpha_signals       : Momentum, reversal, value, quality, stat-arb signal generation
backtester          : Vectorised backtesting engine + walk-forward validation
regime_detection    : HMM, GARCH, rolling-zscore market regime classifiers
execution_model     : Almgren-Chriss impact model, TWAP/VWAP scheduling
"""

from .alpha_signals import (
    combine_signals,
    low_volatility,
    momentum,
    pairs_spread,
    short_term_reversal,
    time_series_momentum,
    z_score_mean_reversion,
)
from .backtester import (
    BacktestConfig,
    BacktestResult,
    VectorisedBacktester,
    WalkForwardValidator,
)
from .execution_model import (
    AlmgrenChrissModel,
    ExecutionParams,
    TWAPScheduler,
    VWAPScheduler,
)
from .portfolio_optimizer import (
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)
from .regime_detection import (
    GARCHRegimeDetector,
    HMMRegimeDetector,
    RegimeAwarePortfolio,
    RollingZScoreRegimeDetector,
)
from .risk_metrics import (
    alpha,
    beta,
    calmar_ratio,
    conditional_value_at_risk,
    information_ratio,
    max_drawdown,
    omega_ratio,
    risk_report,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

__all__ = [
    # risk_metrics
    "value_at_risk",
    "conditional_value_at_risk",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "omega_ratio",
    "beta",
    "alpha",
    "information_ratio",
    "risk_report",
    # portfolio_optimizer
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "BlackLittermanOptimizer",
    # alpha_signals
    "momentum",
    "time_series_momentum",
    "short_term_reversal",
    "low_volatility",
    "z_score_mean_reversion",
    "pairs_spread",
    "combine_signals",
    # backtester
    "BacktestConfig",
    "BacktestResult",
    "VectorisedBacktester",
    "WalkForwardValidator",
    # regime_detection
    "HMMRegimeDetector",
    "GARCHRegimeDetector",
    "RollingZScoreRegimeDetector",
    "RegimeAwarePortfolio",
    # execution_model
    "ExecutionParams",
    "AlmgrenChrissModel",
    "TWAPScheduler",
    "VWAPScheduler",
]
