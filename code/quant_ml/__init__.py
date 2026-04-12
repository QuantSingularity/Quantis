"""
quant_ml — Quantis Machine Learning & Quantitative Finance Package
==================================================================

Subpackages
-----------
data        : Data ingestion, feature engineering, and feature store integration
models      : Time-series models (TFT, LSTM), training, hyperparameter tuning, deployment
monitoring  : Operational metrics collection (CloudWatch)
quant       : Advanced quantitative finance — risk metrics, portfolio optimisation,
              alpha signals, backtesting, regime detection, execution modelling

Quick start
-----------
    from quant_ml.quant import risk_report, MeanVarianceOptimizer, VectorisedBacktester
    from quant_ml.models.train_model import train_model, TemporalFusionTransformer
"""

__version__ = "1.0.0"
