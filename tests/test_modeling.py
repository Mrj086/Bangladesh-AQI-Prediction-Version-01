"""Unit tests for modeling utilities."""

import numpy as np

from src.modeling import regression_metrics


def test_metrics_are_finite_for_normal_predictions():
    metrics = regression_metrics(
        np.array([50.0, 60.0, 70.0]),
        np.array([51.0, 59.0, 72.0]),
    )

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert np.isfinite(metrics["r2"])


def test_metrics_handle_zero_targets():
    metrics = regression_metrics(
        np.array([0.0, 10.0, 20.0]),
        np.array([1.0, 11.0, 18.0]),
    )

    assert np.isfinite(metrics["mape_percent"])
