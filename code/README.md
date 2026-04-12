# Quantis — ML Platform & Quantitative Finance System

## Project Structure

```
quantis_project/
├── backend/                    # FastAPI backend
│   ├── auth/                   # Authentication utilities
│   ├── core/                   # App factory, config, database
│   ├── domain/                 # SQLAlchemy models & Pydantic schemas
│   ├── endpoints/              # API route handlers
│   ├── middleware/             # CORS, auth, logging, error handling
│   ├── services/               # Business logic layer
│   ├── workers/                # Celery background tasks
│   ├── tests/                  # Test suite
│   └── requirements.txt
│
└── quant_ml/                   # ML & Quantitative Finance package (renamed from ml/)
    ├── data/
    │   ├── features/           # Feast feature store integration
    │   └── process_data.py     # Data engine + financial feature engineering
    ├── models/
    │   ├── hyperparameter_tuning/  # Optuna optimisation
    │   ├── model_serving/          # MLflow pyfunc serving
    │   ├── train_model.py          # TFT + LSTM training
    │   ├── mlflow_tracking.py      # Experiment tracking
    │   └── aws_deploy.py           # S3 + SageMaker deployment
    ├── monitoring/
    │   └── metrics_collector.py    # CloudWatch metrics
    └── quant/                  # ★ Advanced Quantitative Finance (new)
        ├── risk_metrics.py         # VaR, CVaR, Sharpe, Sortino, drawdown, alpha/beta
        ├── portfolio_optimizer.py  # Mean-Variance, Risk Parity, Black-Litterman
        ├── alpha_signals.py        # Momentum, value, quality, stat-arb signals
        ├── backtester.py           # Vectorised backtester + walk-forward validation
        ├── regime_detection.py     # HMM, GARCH, rolling-zscore regime classifiers
        └── execution_model.py      # Almgren-Chriss impact, TWAP/VWAP scheduling
```

## Quick Start

### Installation

```bash
cd backend
pip install -r requirements.txt

# Optional quant_ml extras
pip install hmmlearn arch scipy  # for regime detection
pip install feast                 # for feature store
pip install optuna                # for hyperparameter tuning
pip install mlflow                # for experiment tracking
```

### Starting the Backend

```bash
cd backend
uvicorn core.app:app --host 0.0.0.0 --port 8000 --reload
```

The backend starts successfully when you see:

```
INFO: Database initialized successfully
INFO: Uvicorn running on http://0.0.0.0:8000
```

### API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Prometheus Metrics**: http://localhost:8000/metrics

## Environment Configuration

Copy `.env.example` to `.env` and configure:

```env
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET=your-jwt-secret-change-in-production

# Database (defaults to SQLite)
DATABASE_URL=sqlite:///./quantis.db

# Redis (optional — enables Celery background tasks)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0

# Email (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-password
```

## Testing

```bash
# From project root
pytest backend/tests/ -v

# Run specific test module
pytest backend/tests/test_model.py -v
```

## quant_ml — Advanced Quant Finance Usage

```python
from quant_ml.quant import (
    risk_report,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    BlackLittermanOptimizer,
    VectorisedBacktester,
    BacktestConfig,
    WalkForwardValidator,
    HMMRegimeDetector,
    RollingZScoreRegimeDetector,
    momentum,
    low_volatility,
    combine_signals,
    AlmgrenChrissModel,
    ExecutionParams,
)

# Risk metrics
import numpy as np
returns = np.random.randn(252) * 0.01
report = risk_report(returns)
print(report)

# Portfolio optimisation (Mean-Variance)
import pandas as pd
prices = pd.DataFrame(np.random.randn(500, 5).cumsum(axis=0) + 100,
                      columns=list("ABCDE"))
ret_df = prices.pct_change().dropna()
mvo = MeanVarianceOptimizer(ret_df)
result = mvo.max_sharpe()
print(result["weights"])

# Vectorised backtesting
config = BacktestConfig(commission_bps=5, rebalance_freq="W")
bt = VectorisedBacktester(config)
signals = momentum(prices)
bt_result = bt.run(prices, signals)
print(bt_result.performance)

# Regime detection (no extra deps)
from quant_ml.quant import RollingZScoreRegimeDetector
detector = RollingZScoreRegimeDetector(window=63)
regimes = detector.fit_predict(ret_df.iloc[:, 0])

# Execution cost estimation
params = ExecutionParams(
    symbol="AAPL", avg_daily_volume=5_000_000, price=180.0
)
ac = AlmgrenChrissModel(params)
estimate = ac.pre_trade_estimate(order_shares=10_000, horizon_days=5)
print(estimate)
```

## Bug Fixes Applied

| File                                                | Bug                                                             | Fix                                                                     |
| --------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `quant_ml/models/train_model.py`                    | `RuntimeError: Can't call numpy() on Tensor that requires grad` | Added `.detach()` before all `.numpy()` calls                           |
| `quant_ml/models/train_model.py`                    | `train_model(None)` crashed immediately                         | Guard clause converts `None` to `"default"`                             |
| `quant_ml/models/train_model.py`                    | Exploding gradients in LSTM                                     | Added `clip_grad_norm_` + LR scheduler                                  |
| `quant_ml/models/train_model.py`                    | `TemporalFusionTransformer` was just a plain MLP                | Replaced with proper GRN blocks + multi-head attention                  |
| `quant_ml/models/model_serving/serve.py`            | `torch.load()` without `weights_only=True` (pickle RCE risk)    | Added `weights_only=True` with graceful fallback                        |
| `quant_ml/models/mlflow_tracking.py`                | `MlflowException: Run already active` on nested calls           | Added `nested=bool(mlflow.active_run())`                                |
| `quant_ml/models/hyperparameter_tuning/optimize.py` | `direction` not validated before Optuna study creation          | Added explicit validation + `MedianPruner`                              |
| `quant_ml/models/aws_deploy.py`                     | Full local path leaked into S3 key                              | Changed to `os.path.basename(model_path)`                               |
| `quant_ml/data/features/feature_store.py`           | Feast `ValueType` API removed in Feast ≥ 0.30                   | Migrated to `join_keys=["driver_id"]` pattern                           |
| `quant_ml/data/process_data.py`                     | `fit_transform(dask_df)` — sklearn doesn't support Dask         | Added `.compute()` to materialise pandas DataFrame first                |
| `quant_ml/monitoring/metrics_collector.py`          | `boto3.client()` re-created on every metric call                | Client cached at `__init__` time; CloudWatch errors no longer propagate |
| `backend/core/app.py`                               | DB session leaked in `AuditMiddleware` (never closed)           | Wrapped `get_db()` generator in `try/finally`                           |
| `backend/services/financial_service.py`             | `t.amount % Decimal(...)` may fail if `amount` is `float`       | Wrapped all comparisons with `Decimal(str(t.amount))`                   |
| `backend/services/prediction_service.py`            | `batch_predict` silently dropped errors with bare `continue`    | Added error counter + warning log at end of batch                       |
| All test files                                      | Imported from `ml.` (old package name)                          | Updated to `quant_ml.`                                                  |
