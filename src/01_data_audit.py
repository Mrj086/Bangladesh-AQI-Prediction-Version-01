"""Create a reproducible data-quality audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import METRICS, ensure_directories, read_dataset, save_json


POLLUTANTS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


def build_audit(frame: pd.DataFrame) -> dict:
    """Build structural, temporal, missingness, and outlier diagnostics."""
    numeric = frame.select_dtypes(include="number")
    missing = frame.isna().sum().sort_values(ascending=False)

    iqr_outliers = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            iqr_outliers[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        iqr_outliers[column] = int(mask.sum())

    city_time = frame[["city_name", "datetime"]].dropna().copy()
    city_time = city_time.sort_values(["city_name", "datetime"])
    deltas = city_time.groupby("city_name")["datetime"].diff().dropna()

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "city_count": int(frame["city_name"].nunique()),
        "cities": sorted(frame["city_name"].dropna().unique().tolist()),
        "datetime_min": frame["datetime"].min(),
        "datetime_max": frame["datetime"].max(),
        "invalid_timestamps": int(frame["datetime"].isna().sum()),
        "duplicate_city_timestamps": int(
            frame.duplicated(["city_name", "datetime"]).sum()
        ),
        "missing_values": {k: int(v) for k, v in missing.items()},
        "missing_percent": {
            k: float(v / len(frame) * 100) for k, v in missing.items()
        },
        "iqr_outlier_counts": iqr_outliers,
        "negative_pollutant_counts": {
            column: int((frame[column] < 0).sum())
            for column in POLLUTANTS
        },
        "hourly_delta_count": int((deltas == pd.Timedelta(hours=1)).sum()),
        "non_hourly_delta_count": int(
            (deltas != pd.Timedelta(hours=1)).sum()
        ),
        "summary": frame.describe(include="all").to_dict(),
    }


def main() -> None:
    """Run the audit CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    ensure_directories()
    frame = read_dataset(args.input)
    report = build_audit(frame)
    output = METRICS / "data_audit.json"
    save_json(report, output)
    print(f"Audit written to: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
