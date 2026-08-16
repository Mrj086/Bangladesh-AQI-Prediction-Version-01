"""Reusable modeling components for leakage-safe AQI forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns using thresholds learned from training data."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).astype(float)
        self.lower_bounds_ = frame.quantile(self.lower)
        self.upper_bounds_ = frame.quantile(self.upper)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).astype(float)
        clipped = frame.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis="columns",
        )
        return clipped.to_numpy()


class ColumnNameSelector(BaseEstimator, TransformerMixin):
    """Preserve DataFrame columns while enabling sklearn transformations."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        return X[self.columns_]


@dataclass
class ModelSpec:
    """Named model configuration."""

    name: str
    estimator: object


def get_feature_columns(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Return numeric and categorical predictor columns."""
    excluded = {
        target,
        "datetime",
        "city_id",
    }

    features = [c for c in frame.columns if c not in excluded]
    numeric = frame[features].select_dtypes(include=np.number).columns.tolist()
    numeric = [c for c in numeric if frame[c].notna().any()]
    categorical = [
        c for c in features
        if c not in numeric and frame[c].notna().any()
    ]
    return numeric, categorical


def build_preprocessor(
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> ColumnTransformer:
    """Build train-fitted imputation, clipping, scaling, and encoding."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clipper", QuantileClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_specs(random_state: int = 42) -> list[ModelSpec]:
    """Return the model benchmark suite."""
    return [
        ModelSpec(
            "ridge",
            Ridge(alpha=10.0),
        ),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                max_features=0.7,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=63,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=10,
                random_state=random_state,
            ),
        ),
    ]


def make_pipeline(
    model: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Create a complete estimator whose preprocessing is fit only on training data."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_columns, categorical_columns),
            ),
            ("model", model),
        ]
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate robust regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    nonzero = np.abs(y_true) > 1e-9
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))
        * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(mape),
        "median_absolute_error": float(
            np.median(np.abs(y_true - y_pred))
        ),
        "explained_variance": float(
            explained_variance_score(y_true, y_pred)
        ),
    }
