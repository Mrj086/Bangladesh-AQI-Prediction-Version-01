"""Clean the raw dataset and build leakage-safe forecasting features."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import DATA_PROCESSED, ensure_directories, read_dataset


POLLUTANTS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

LAGS = [1, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 168]


def clean_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic data-quality rules before feature engineering."""
    frame = frame.copy()
    frame = frame.dropna(subset=["datetime", "city_name"])
    frame = frame.drop_duplicates(["city_name", "datetime"])

    for column in POLLUTANTS + ["aqi"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid = frame[POLLUTANTS] < 0
    frame[POLLUTANTS] = frame[POLLUTANTS].mask(invalid)

    frame.loc[frame["aqi"] < 0, "aqi"] = np.nan

    frame = frame.sort_values(["city_name", "datetime"]).reset_index(drop=True)
    return frame


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical time features."""
    result = frame.copy()
    dt = result["datetime"]

    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["day_of_month"] = dt.dt.day
    result["month"] = dt.dt.month
    result["quarter"] = dt.dt.quarter
    result["day_of_year"] = dt.dt.dayofyear
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["is_rush_hour"] = result["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["year_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["year_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)

    return result


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add autoregressive features using only past observations."""
    result = frame.copy()
    grouped = result.groupby("city_name", sort=False)

    for column in POLLUTANTS + ["aqi"]:
        for lag in LAGS:
            result[f"{column}_lag_{lag}h"] = grouped[column].shift(lag)

    return result


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add historical rolling statistics without using the current value."""
    result = frame.copy()

    for column in POLLUTANTS + ["aqi"]:
        shifted = result.groupby("city_name", sort=False)[column].shift(1)
        for window in ROLLING_WINDOWS:
            grouped_shifted = shifted.groupby(result["city_name"], sort=False)
            result[f"{column}_roll_{window}h_mean"] = grouped_shifted.transform(
                lambda s: s.rolling(window, min_periods=2).mean()
            )
            result[f"{column}_roll_{window}h_std"] = grouped_shifted.transform(
                lambda s: s.rolling(window, min_periods=2).std()
            )

    return result


def add_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add first differences and safe percentage changes."""
    result = frame.copy()
    grouped = result.groupby("city_name", sort=False)

    for column in POLLUTANTS + ["aqi"]:
        lag = grouped[column].shift(1)
        result[f"{column}_diff_1h"] = result[column] - lag
        denominator = lag.abs().clip(lower=1e-3)
        result[f"{column}_pct_change_1h"] = (result[column] - lag) / denominator

    return result


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Create an exact next-hour target rather than next-row target."""
    result = frame.copy()
    target = result[["city_name", "datetime", "aqi"]].copy()
    target["datetime"] = target["datetime"] - pd.Timedelta(hours=1)
    target = target.rename(columns={"aqi": "target_aqi_next_hour"})

    result = result.merge(
        target,
        on=["city_name", "datetime"],
        how="left",
        validate="one_to_one",
    )
    return result


def build_dataset(input_path: str) -> pd.DataFrame:
    """Run the complete deterministic feature-engineering pipeline."""
    frame = clean_raw(read_dataset(input_path))
    frame = add_time_features(frame)
    frame = add_lag_features(frame)
    frame = add_rolling_features(frame)
    frame = add_dynamics(frame)
    frame = add_target(frame)

    frame["city_lat"] = frame["lat"]
    frame["city_lon"] = frame["lon"]

    frame = frame.dropna(subset=["target_aqi_next_hour"]).reset_index(drop=True)
    return frame


def main() -> None:
    """Run feature engineering from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    ensure_directories()
    frame = build_dataset(args.input)
    output = DATA_PROCESSED / "model_dataset.parquet"
    frame.to_parquet(output, index=False)
    print(f"Feature dataset written to: {output}")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {frame.shape[1]:,}")


if __name__ == "__main__":
    main()
