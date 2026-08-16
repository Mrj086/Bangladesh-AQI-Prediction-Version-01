# app/api.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.deep_learning import create_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"

SUPPORTED_MODELS = {
    "lstm",
    "gru",
    "tcn",
    "transformer",
}

DEFAULT_MODEL = "transformer"

app = FastAPI(
    title="Bangladesh AQI Forecasting API",
    version="3.0.0",
)


class Observation(BaseModel):
    """One hourly observation used for next-hour AQI prediction."""

    datetime: Optional[datetime] = None
    aqi: Optional[float] = None
    pm10: Optional[float] = None
    pm2_5: Optional[float] = None
    carbon_monoxide: Optional[float] = None
    carbon_dioxide: Optional[float] = None
    nitrogen_dioxide: Optional[float] = None
    sulphur_dioxide: Optional[float] = None
    ozone: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    day_of_week_sin: Optional[float] = None
    day_of_week_cos: Optional[float] = None
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None
    is_weekend: Optional[float] = None


class ForecastRequest(BaseModel):
    """Historical observations used for next-hour AQI prediction."""

    observations: list[Observation] = Field(
        ...,
        min_length=1,
        description="Historical hourly observations.",
    )

    model: str = Field(
        default=DEFAULT_MODEL,
        description="One of: lstm, gru, tcn, transformer.",
    )


def get_model_path(model_name: str) -> Path:
    """Return the checkpoint path for a supported model."""
    model_name = model_name.lower().strip()

    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported model '{model_name}'. "
                f"Choose one of: {', '.join(sorted(SUPPORTED_MODELS))}."
            ),
        )

    return MODEL_DIR / f"{model_name}.pt"


def load_checkpoint(
    model_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a PyTorch checkpoint and its preprocessing metadata."""
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artifact not found: {model_path}. "
                "Train the deep-learning pipeline first."
            ),
        )

    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load model checkpoint: {exc}",
        ) from exc

    required_keys = {
        "model_name",
        "window",
        "feature_columns",
        "mean",
        "scale",
        "state_dict",
    }

    missing = required_keys.difference(checkpoint)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model checkpoint is missing required metadata: "
                + ", ".join(sorted(missing))
            ),
        )

    try:
        feature_columns = list(checkpoint["feature_columns"])
        window = int(checkpoint["window"])

        model = create_model(
            checkpoint["model_name"],
            len(feature_columns),
            window,
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to initialize model: {exc}",
        ) from exc

    return model, checkpoint


def prepare_features(
    observations: list[Observation],
    feature_columns: list[str],
    window: int,
) -> tuple[np.ndarray, float]:
    """Prepare a model window and preserve the raw AQI baseline."""
    if len(observations) < window:
        raise HTTPException(
            status_code=422,
            detail=(
                f"At least {window} observations are required "
                f"for this model."
            ),
        )

    frame = pd.DataFrame(
        [
            observation.model_dump(exclude_none=False)
            for observation in observations[-window:]
        ]
    )

    if "carbon_dioxide_missing" in feature_columns:
        if "carbon_dioxide" not in frame.columns:
            frame["carbon_dioxide"] = np.nan

        frame["carbon_dioxide_missing"] = (
            frame["carbon_dioxide"]
            .isna()
            .astype(np.float32)
        )

    missing_columns = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=(
                "Missing required model features: "
                + ", ".join(missing_columns)
            ),
        )

    if "aqi" not in frame.columns:
        raise HTTPException(
            status_code=422,
            detail="AQI is required because the models use the latest AQI as baseline.",
        )

    raw_aqi = pd.to_numeric(
        frame["aqi"],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if raw_aqi.isna().all():
        raise HTTPException(
            status_code=422,
            detail="At least one valid AQI value is required.",
        )

    raw_aqi = raw_aqi.ffill().bfill()

    if raw_aqi.isna().any():
        raise HTTPException(
            status_code=422,
            detail="Unable to determine the latest AQI baseline.",
        )

    baseline_aqi = float(raw_aqi.iloc[-1])

    values = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if "carbon_dioxide" in values.columns:
        values["carbon_dioxide"] = values["carbon_dioxide"].fillna(0.0)

    values = values.ffill().bfill()

    unresolved = [
        column
        for column in feature_columns
        if values[column].isna().any()
    ]

    if unresolved:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to fill missing values for: "
                + ", ".join(unresolved)
            ),
        )

    return (
        values.to_numpy(dtype=np.float32),
        baseline_aqi,
    )


def predict_aqi(
    model: torch.nn.Module,
    values: np.ndarray,
    baseline_aqi: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Run inference using the raw AQI persistence baseline."""
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature window, got shape {values.shape}."
        )

    if values.shape[1] != len(mean):
        raise ValueError(
            "Feature count does not match checkpoint scaler."
        )

    if len(scale) != values.shape[1]:
        raise ValueError(
            "Scale length does not match feature count."
        )

    if not np.isfinite(baseline_aqi):
        raise ValueError(
            "Baseline AQI must be finite."
        )

    scale = np.where(
        np.isfinite(scale) & (scale > 1e-8),
        scale,
        1.0,
    )

    scaled = (
        (values - mean) / scale
    ).astype(np.float32)

    tensor = torch.from_numpy(
        scaled
    ).unsqueeze(0)

    baseline = torch.tensor(
        [baseline_aqi],
        dtype=torch.float32,
    )

    try:
        with torch.no_grad():
            prediction = model(
                tensor,
                baseline,
            )
    except Exception as exc:
        raise ValueError(
            f"Model inference failed: {exc}"
        ) from exc

    value = float(
        prediction.squeeze().cpu().item()
    )

    if not np.isfinite(value):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    return float(
        np.clip(
            value,
            0.0,
            500.0,
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {
        "status": "ok",
    }


@app.get("/models")
def models() -> dict[str, Any]:
    """Return available model checkpoints."""
    available = []

    for model_name in sorted(SUPPORTED_MODELS):
        path = get_model_path(model_name)

        available.append(
            {
                "model": model_name,
                "available": path.exists(),
                "checkpoint": str(path),
            }
        )

    return {
        "default_model": DEFAULT_MODEL,
        "models": available,
    }


@app.get("/model-info")
def model_info(
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Return metadata for the selected model."""
    model_path = get_model_path(model)

    _, checkpoint = load_checkpoint(
        model_path
    )

    return {
        "model": checkpoint["model_name"],
        "window_hours": int(checkpoint["window"]),
        "feature_columns": list(
            checkpoint["feature_columns"]
        ),
        "device": "cpu",
        "checkpoint": str(model_path),
    }


@app.post("/predict")
def predict(
    request: ForecastRequest,
) -> dict[str, Any]:
    """Predict next-hour AQI from historical observations."""
    model_path = get_model_path(
        request.model
    )

    model, checkpoint = load_checkpoint(
        model_path
    )

    feature_columns = list(
        checkpoint["feature_columns"]
    )

    window = int(
        checkpoint["window"]
    )

    values, baseline_aqi = prepare_features(
        request.observations,
        feature_columns,
        window,
    )

    try:
        prediction = predict_aqi(
            model=model,
            values=values,
            baseline_aqi=baseline_aqi,
            mean=np.asarray(
                checkpoint["mean"],
                dtype=np.float32,
            ),
            scale=np.asarray(
                checkpoint["scale"],
                dtype=np.float32,
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {
        "model": checkpoint["model_name"],
        "window_hours": window,
        "baseline_aqi": baseline_aqi,
        "predicted_next_hour_aqi": prediction,
        "change_from_baseline": prediction - baseline_aqi,
        "feature_columns": feature_columns,
    }