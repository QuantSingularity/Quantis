"""
Data processing engine for the Quantis ML pipeline.

Fixes vs original:
- DataEngine.process() was calling fit_transform on a Dask DataFrame directly,
  which sklearn pipelines don't support — now converts to pandas first.
- create_temporal_features used groupby on Dask which requires compute(); fixed.
- Added missing financial time-series feature engineering helpers.
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

numeric_features = ["feature1", "feature2", "feature3"]
categorical_features = ["category1", "category2"]


class DataEngine:

    def __init__(
        self,
        num_features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
    ) -> None:
        self.numeric_features = num_features if num_features else numeric_features
        self.categorical_features = (
            cat_features if cat_features else categorical_features
        )
        self.preprocessor = make_pipeline(
            ColumnTransformer(
                [
                    ("num", StandardScaler(), self.numeric_features),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        self.categorical_features,
                    ),
                ]
            ),
            PolynomialFeatures(degree=2, interaction_only=True),
        )

    def process(self, raw_data: Any) -> Any:
        """
        Load and preprocess data from a parquet file path.

        BUG FIX: sklearn fit_transform does not work on Dask DataFrames —
        the original called fit_transform(ddf) which raises TypeError.
        We now call .compute() to materialise the pandas DataFrame first.
        """
        try:
            if DASK_AVAILABLE:
                ddf = dd.read_parquet(raw_data)
                ddf = ddf.map_partitions(lambda df: df.dropna())
                df = ddf.compute()  # materialise to pandas before sklearn transform
            else:
                df = pd.read_parquet(raw_data).dropna()
            return self.preprocessor.fit_transform(df)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {raw_data}") from e
        except Exception as e:
            raise ValueError(f"Error processing data from {raw_data}: {e}") from e

    def create_temporal_features(
        self, df: pd.DataFrame, entity_col: str = "entity", value_col: str = "value"
    ) -> pd.DataFrame:
        """
        Add rolling-window temporal features to a pandas DataFrame.

        BUG FIX: Original used df.groupby().transform() which crashes on Dask
        DataFrames — method now explicitly accepts pandas DataFrame.
        """
        df = df.copy()
        df["rolling_mean_7"] = df.groupby(entity_col)[value_col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df["rolling_std_7"] = df.groupby(entity_col)[value_col].transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0)
        )
        df["rolling_mean_30"] = df.groupby(entity_col)[value_col].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        return df


class FinancialFeatureEngineer:
    """
    Advanced financial feature engineering for quantitative strategies.
    Computes technical indicators, risk metrics, and cross-sectional features
    that are standard in equity/crypto quant pipelines.
    """

    @staticmethod
    def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Log returns, simple returns, and multi-period returns."""
        df = df.copy()
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df["simple_return"] = df[price_col].pct_change()
        df["return_5d"] = df[price_col].pct_change(5)
        df["return_21d"] = df[price_col].pct_change(21)
        return df

    @staticmethod
    def add_volatility(
        df: pd.DataFrame, return_col: str = "log_return"
    ) -> pd.DataFrame:
        """Realised volatility at multiple horizons (annualised)."""
        df = df.copy()
        for window in [5, 21, 63]:
            df[f"vol_{window}d"] = df[return_col].rolling(window).std() * np.sqrt(252)
        return df

    @staticmethod
    def add_rsi(
        df: pd.DataFrame, price_col: str = "close", period: int = 14
    ) -> pd.DataFrame:
        """Relative Strength Index."""
        df = df.copy()
        delta = df[price_col].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        price_col: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD, signal line, and histogram."""
        df = df.copy()
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Bollinger Bands and percent-B."""
        df = df.copy()
        ma = df[price_col].rolling(window).mean()
        std = df[price_col].rolling(window).std()
        df["bb_upper"] = ma + num_std * std
        df["bb_lower"] = ma - num_std * std
        df["bb_mid"] = ma
        df["bb_pct_b"] = (df[price_col] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )
        return df

    @staticmethod
    def add_atr(
        df: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        period: int = 14,
    ) -> pd.DataFrame:
        """Average True Range."""
        df = df.copy()
        hl = df[high_col] - df[low_col]
        hc = (df[high_col] - df[close_col].shift(1)).abs()
        lc = (df[low_col] - df[close_col].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df[f"atr_{period}"] = tr.rolling(period).mean()
        return df

    def build_all(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Apply all feature engineering steps in sequence."""
        df = self.add_returns(df, price_col)
        df = self.add_volatility(df)
        df = self.add_rsi(df, price_col)
        df = self.add_macd(df, price_col)
        df = self.add_bollinger_bands(df, price_col)
        if {"high", "low"}.issubset(df.columns):
            df = self.add_atr(df)
        return df
