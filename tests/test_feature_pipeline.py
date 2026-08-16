"""Unit tests for the feature engineering layer."""

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

feature_module = importlib.import_module("src.02_build_features")
add_target = feature_module.add_target


def test_exact_next_hour_target():
    frame = pd.DataFrame(
        {
            "city_name": ["A", "A", "A"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 01:00",
                    "2025-01-01 02:00",
                ]
            ),
            "aqi": [50.0, 60.0, 70.0],
        }
    )

    result = add_target(frame)

    assert result.loc[0, "target_aqi_next_hour"] == 60.0
    assert result.loc[1, "target_aqi_next_hour"] == 70.0
    assert pd.isna(result.loc[2, "target_aqi_next_hour"])
