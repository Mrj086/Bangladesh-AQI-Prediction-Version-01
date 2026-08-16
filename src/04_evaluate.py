"""Generate publication-oriented model evaluation tables and figures."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, METRICS, ensure_directories
from modeling import regression_metrics


def plot_actual_vs_predicted(frame: pd.DataFrame) -> None:
    """Save actual-vs-predicted diagnostic."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        frame["target_aqi_next_hour"],
        frame["prediction"],
        s=8,
        alpha=0.25,
    )
    limits = [
        frame["target_aqi_next_hour"].min(),
        frame["target_aqi_next_hour"].max(),
    ]
    ax.plot(limits, limits, linewidth=2)
    ax.set_xlabel("Actual next-hour AQI")
    ax.set_ylabel("Predicted next-hour AQI")
    ax.set_title("Actual vs Predicted AQI")
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)


def plot_residuals(frame: pd.DataFrame) -> None:
    """Save residual distribution."""
    residuals = frame["residual"].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=80)
    ax.axvline(0, linewidth=2)
    ax.set_xlabel("Residual: actual - prediction")
    ax.set_ylabel("Frequency")
    ax.set_title("Forecast Residual Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_distribution.png", dpi=180)
    plt.close(fig)


def city_error_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate city-level performance."""
    rows = []
    for city, group in frame.groupby("city_name"):
        metrics = regression_metrics(
            group["target_aqi_next_hour"],
            group["prediction"],
        )
        rows.append({"city_name": city, **metrics})
    return pd.DataFrame(rows).sort_values("rmse")


def plot_city_rmse(city_metrics: pd.DataFrame) -> None:
    """Save city-level RMSE comparison."""
    plot_data = city_metrics.sort_values("rmse", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(plot_data["city_name"], plot_data["rmse"])
    ax.set_xlabel("RMSE")
    ax.set_ylabel("City")
    ax.set_title("Next-Hour AQI RMSE by City")
    fig.tight_layout()
    fig.savefig(FIGURES / "city_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run evaluation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    ensure_directories()
    frame = pd.read_parquet(args.input)

    required = {
        "city_name",
        "datetime",
        "target_aqi_next_hour",
        "prediction",
        "residual",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    city_metrics = city_error_table(frame)
    city_metrics.to_csv(METRICS / "city_metrics.csv", index=False)

    plot_actual_vs_predicted(frame)
    plot_residuals(frame)
    plot_city_rmse(city_metrics)

    hourly = (
        frame.assign(hour=frame["datetime"].dt.hour)
        .groupby("hour")
        .apply(
            lambda group: pd.Series(
                regression_metrics(
                    group["target_aqi_next_hour"],
                    group["prediction"],
                )
            ),
            include_groups=False,
        )
        .reset_index()
    )
    hourly.to_csv(METRICS / "hourly_metrics.csv", index=False)

    print("Evaluation artifacts written to reports/.")


if __name__ == "__main__":
    main()
