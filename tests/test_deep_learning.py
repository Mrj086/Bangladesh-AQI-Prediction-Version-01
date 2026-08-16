"""Tests for deep-learning sequence preparation and model shapes."""

import numpy as np
import pandas as pd
import torch

from src.deep_learning import (
    MODEL_NAMES,
    build_sequence_data,
    create_model,
)


def make_frame(include_co2: bool = False) -> pd.DataFrame:
    """Create a small hourly multi-city fixture."""
    rows = []
    for city in ["A", "B"]:
        times = pd.date_range("2025-01-01", periods=80, freq="h")
        for i, timestamp in enumerate(times):
            row = {
                "city_name": city,
                "datetime": timestamp,
                "aqi": 40 + i % 30,
                "pm10": 10 + i,
                "pm2_5": 5 + i / 2,
                "carbon_monoxide": 200 + i,
                "nitrogen_dioxide": 10 + i / 10,
                "sulphur_dioxide": 2 + i / 20,
                "ozone": 20 + i,
                "lat": 23.7,
                "lon": 90.4,
                "hour_sin": np.sin(2 * np.pi * i / 24),
                "hour_cos": np.cos(2 * np.pi * i / 24),
                "day_of_week_sin": 0.0,
                "day_of_week_cos": 1.0,
                "month_sin": 0.0,
                "month_cos": 1.0,
                "is_weekend": 0,
                "target_aqi_next_hour": 41 + i % 30,
            }
            if include_co2:
                row["carbon_dioxide"] = np.nan if i % 10 == 0 else 400 + i
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_split_is_nonempty_and_leakage_safe():
    """Chronological splits should not overlap."""
    data, train, val, test = build_sequence_data(
        make_frame(),
        window=24,
        train_fraction=0.60,
        val_fraction=0.20,
    )

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert "aqi" in data.feature_columns


def test_partial_co2_keeps_value_and_missing_indicator():
    """Partially observed CO2 should retain both the value and missingness signal."""
    data, *_ = build_sequence_data(
        make_frame(include_co2=True),
        window=24,
        train_fraction=0.60,
        val_fraction=0.20,
    )

    assert "carbon_dioxide" in data.feature_columns
    assert "carbon_dioxide_missing" in data.feature_columns


def test_all_deep_models_return_one_prediction():
    """Every architecture should accept a sequence and persistence baseline."""
    for name in MODEL_NAMES:
        model = create_model(name, input_size=8, window=24)
        output = model(
            torch.randn(2, 24, 8),
            torch.tensor([50.0, 55.0]),
        )
        assert output.shape == (2,)


def test_residual_models_start_at_persistence():
    """Zero-initialized residual heads should reproduce persistence before training."""
    baseline = torch.tensor([50.0, 75.0])
    inputs = torch.randn(2, 24, 8)

    for name in MODEL_NAMES:
        model = create_model(name, input_size=8, window=24)
        output = model(inputs, baseline)
        assert torch.allclose(output, baseline, atol=1e-5)
