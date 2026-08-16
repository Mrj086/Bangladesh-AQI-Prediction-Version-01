"""Train, cross-validate, compare, and persist AQI forecasting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from common import METRICS, MODELS, ensure_directories
from modeling import (
    get_feature_columns,
    make_pipeline,
    model_specs,
    regression_metrics,
)


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by timestamp so the final period is never used for development."""
    timestamps = np.sort(frame["datetime"].dropna().unique())
    boundary_index = int(len(timestamps) * (1 - test_fraction))
    boundary = timestamps[boundary_index]

    development = frame[frame["datetime"] < boundary].copy()
    test = frame[frame["datetime"] >= boundary].copy()
    return development, test


def fit_and_cv(
    frame: pd.DataFrame,
    target: str,
    random_state: int,
    cv_splits: int,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run rolling-origin CV and fit each candidate on development data."""
    frame = frame.sort_values(["datetime", "city_name"]).reset_index(drop=True)
    numeric, categorical = get_feature_columns(frame, target)
    x = frame[numeric + categorical]
    y = frame[target]

    unique_times = np.sort(frame["datetime"].dropna().unique())
    if len(unique_times) <= cv_splits:
        raise ValueError("Not enough unique timestamps for requested CV splits.")

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    results = []
    fitted = {}

    for spec in model_specs(random_state):
        fold_metrics = []

        for fold, (train_time_idx, valid_time_idx) in enumerate(
            tscv.split(unique_times),
            start=1,
        ):
            train_times = unique_times[train_time_idx]
            valid_times = unique_times[valid_time_idx]
            train_mask = frame["datetime"].isin(train_times).to_numpy()
            valid_mask = frame["datetime"].isin(valid_times).to_numpy()

            model = make_pipeline(spec.estimator, numeric, categorical)
            model.fit(x.loc[train_mask], y.loc[train_mask])
            pred = model.predict(x.loc[valid_mask])
            metrics = regression_metrics(y.loc[valid_mask], pred)
            metrics["fold"] = fold
            fold_metrics.append(metrics)

        fold_frame = pd.DataFrame(fold_metrics)
        aggregate = {
            "model": spec.name,
            **{
                f"cv_mean_{column}": float(fold_frame[column].mean())
                for column in [
                    "mae",
                    "rmse",
                    "r2",
                    "mape_percent",
                    "median_absolute_error",
                    "explained_variance",
                ]
            },
        }
        results.append(aggregate)

        final_model = make_pipeline(spec.estimator, numeric, categorical)
        final_model.fit(x, y)
        fitted[spec.name] = final_model

    return pd.DataFrame(results), fitted, {
        "numeric_features": numeric,
        "categorical_features": categorical,
    }


def main() -> None:
    """Run the training CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.sample_fraction <= 1:
        raise ValueError("sample-fraction must be in (0, 1].")

    ensure_directories()
    frame = pd.read_parquet(args.input)

    if args.sample_fraction < 1:
        frame = (
            frame.sample(frac=args.sample_fraction, random_state=args.random_state)
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    target = "target_aqi_next_hour"
    development, test = chronological_split(frame, args.test_fraction)

    cv_results, fitted_models, metadata = fit_and_cv(
        development,
        target,
        args.random_state,
        args.cv_splits,
    )

    cv_results = cv_results.sort_values("cv_mean_rmse").reset_index(drop=True)
    cv_results.to_csv(METRICS / "cross_validation_results.csv", index=False)

    best_name = cv_results.iloc[0]["model"]
    best_model = fitted_models[best_name]

    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]

    x_test = test[numeric + categorical]
    y_test = test[target]
    test_pred = best_model.predict(x_test)

    final_metrics = regression_metrics(y_test, test_pred)

    baseline = test.groupby("city_name", sort=False)["aqi"].shift(0)
    baseline = baseline.fillna(test["target_aqi_next_hour"].median())
    baseline_metrics = regression_metrics(y_test, baseline)

    payload = {
        "best_model": best_name,
        "random_state": args.random_state,
        "sample_fraction": args.sample_fraction,
        "development_rows": len(development),
        "test_rows": len(test),
        "development_start": str(development["datetime"].min()),
        "development_end": str(development["datetime"].max()),
        "test_start": str(test["datetime"].min()),
        "test_end": str(test["datetime"].max()),
        "final_test_metrics": final_metrics,
        "persistence_baseline_metrics": baseline_metrics,
        "feature_counts": {
            "numeric": len(numeric),
            "categorical": len(categorical),
        },
    }

    (METRICS / "final_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    joblib.dump(best_model, MODELS / "best_model.joblib")

    prediction_frame = test[
        ["city_name", "datetime", "aqi", target]
    ].copy()
    prediction_frame["prediction"] = test_pred
    prediction_frame["residual"] = (
        prediction_frame[target] - prediction_frame["prediction"]
    )
    prediction_frame.to_parquet(
        METRICS / "test_predictions.parquet",
        index=False,
    )

    print(cv_results.to_string(index=False))
    print("\nBest model:", best_name)
    print("Final test metrics:", final_metrics)
    print("Artifacts written under models/ and reports/metrics/.")


if __name__ == "__main__":
    main()
