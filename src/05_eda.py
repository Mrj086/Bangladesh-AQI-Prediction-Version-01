"""Create advanced exploratory visualizations from the raw dataset."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import FIGURES, ensure_directories, read_dataset


def plot_missingness(frame: pd.DataFrame) -> None:
    """Plot missing-value percentages."""
    missing = frame.isna().mean().mul(100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    missing.plot.bar(ax=ax)
    ax.set_ylabel("Missing values (%)")
    ax.set_title("Data Missingness Profile")
    fig.tight_layout()
    fig.savefig(FIGURES / "missingness.png", dpi=180)
    plt.close(fig)


def plot_correlation(frame: pd.DataFrame) -> None:
    """Plot pollutant/AQI correlation matrix."""
    columns = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "carbon_dioxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "aqi",
    ]
    corr = frame[columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="vlag", center=0)
    ax.set_title("Pollutant and AQI Correlation Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_matrix.png", dpi=180)
    plt.close(fig)


def plot_city_distribution(frame: pd.DataFrame) -> None:
    """Plot AQI distributions by city."""
    top_cities = (
        frame.groupby("city_name")["aqi"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.boxplot(
        data=frame,
        y="city_name",
        x="aqi",
        order=top_cities,
        showfliers=False,
        ax=ax,
    )
    ax.set_title("AQI Distribution by City")
    fig.tight_layout()
    fig.savefig(FIGURES / "city_aqi_distribution.png", dpi=180)
    plt.close(fig)


def plot_hour_month_heatmap(frame: pd.DataFrame) -> None:
    """Plot median AQI by hour and month."""
    work = frame.copy()
    work["hour"] = work["datetime"].dt.hour
    work["month"] = work["datetime"].dt.month

    pivot = work.pivot_table(
        index="hour",
        columns="month",
        values="aqi",
        aggfunc="median",
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, cmap="magma", ax=ax)
    ax.set_title("Median AQI by Hour and Month")
    fig.tight_layout()
    fig.savefig(FIGURES / "aqi_hour_month_heatmap.png", dpi=180)
    plt.close(fig)


def plot_geography(frame: pd.DataFrame) -> None:
    """Plot city coordinates with median AQI as marker size."""
    city = (
        frame.groupby("city_name")
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            median_aqi=("aqi", "median"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        city["lon"],
        city["lat"],
        s=np.maximum(city["median_aqi"], 1) * 2,
        alpha=0.7,
    )

    for _, row in city.iterrows():
        ax.annotate(
            row["city_name"],
            (row["lon"], row["lat"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Bangladesh City AQI Spatial Overview")
    fig.tight_layout()
    fig.savefig(FIGURES / "city_geography.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run EDA CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    ensure_directories()
    frame = read_dataset(args.input)

    plot_missingness(frame)
    plot_correlation(frame)
    plot_city_distribution(frame)
    plot_hour_month_heatmap(frame)
    plot_geography(frame)

    print("EDA figures written to reports/figures/.")


if __name__ == "__main__":
    main()
