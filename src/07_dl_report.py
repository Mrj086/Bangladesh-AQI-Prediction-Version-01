"""Create publication-oriented deep-learning benchmark figures."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES, METRICS, ensure_directories


def plot_training_curves(history: dict) -> None:
    """Plot train/validation Huber loss for every deep model."""
    for model, values in history.items():
        epochs = range(1, len(values["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(epochs, values["train_loss"], label="Train")
        ax.plot(epochs, values["val_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber loss")
        ax.set_title(f"{model.upper()} Training Curve")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / f"dl_{model}_training_curve.png", dpi=220)
        plt.close(fig)


def plot_benchmark(results: pd.DataFrame) -> None:
    """Plot deep-learning RMSE and R² benchmark."""
    ordered = results.sort_values("rmse")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ordered["model"], ordered["rmse"])
    ax.set_ylabel("Test RMSE")
    ax.set_title("Deep-Learning AQI Forecasting Benchmark")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "dl_model_rmse_comparison.png", dpi=220)
    plt.close(fig)


def plot_ensemble_predictions(predictions: pd.DataFrame) -> None:
    """Plot a recent actual-vs-ensemble segment."""
    frame = predictions.sort_values("datetime").tail(1000)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(frame["datetime"], frame["actual_aqi"], label="Actual")
    ax.plot(
        frame["datetime"],
        frame["ensemble_prediction"],
        label="Weighted ensemble",
        linewidth=1.5,
    )
    ax.set_xlabel("Datetime")
    ax.set_ylabel("AQI")
    ax.set_title("Recent Deep-Learning Ensemble Forecasts")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "dl_ensemble_recent_forecast.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Generate deep-learning figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=str(METRICS / "deep_learning_results.csv"),
    )
    args = parser.parse_args()

    ensure_directories()

    results = pd.read_csv(args.results)
    history_path = METRICS / "deep_learning_history.json"
    predictions_path = METRICS / "deep_learning_predictions.parquet"

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        plot_training_curves(history)

    plot_benchmark(results)

    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        plot_ensemble_predictions(predictions)

    print("Deep-learning figures written to reports/figures/.")


if __name__ == "__main__":
    main()
