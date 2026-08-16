"""Shared utilities for the Bangladesh AQI forecasting project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"


def ensure_directories() -> None:
    """Create project output directories."""
    for path in (DATA_RAW, DATA_PROCESSED, MODELS, FIGURES, METRICS):
        path.mkdir(parents=True, exist_ok=True)


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read and normalize the raw AQI dataset."""
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.sort_values(["city_name", "datetime"]).reset_index(drop=True)
    return frame


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Persist JSON with NumPy/Pandas-safe serialization."""
    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, default=convert),
        encoding="utf-8",
    )
