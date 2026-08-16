"""Deep-learning models and leakage-safe sequence utilities for AQI forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


DEFAULT_FEATURES = [
    "aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "carbon_dioxide_missing",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "lat",
    "lon",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

MODEL_NAMES = ("lstm", "gru", "tcn", "transformer")


@dataclass
class SequenceData:
    """In-memory representation of sequence forecasting data."""

    features: np.ndarray
    targets: np.ndarray
    baseline_aqi: np.ndarray
    timestamps: np.ndarray
    cities: np.ndarray
    feature_columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    window: int
    baseline_feature_index: int


class SequenceDataset(Dataset):
    """Lazy sequence dataset backed by a dense feature matrix."""

    def __init__(self, data: SequenceData, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        end = int(self.indices[position])
        start = end - self.data.window + 1
        x = self.data.features[start : end + 1]
        y = self.data.targets[end]
        baseline = self.data.baseline_aqi[end]
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(baseline, dtype=torch.float32),
        )


def available_features(frame: pd.DataFrame) -> list[str]:
    """Select informative numeric predictors with explicit CO2 handling."""
    features = [
        column for column in DEFAULT_FEATURES if column in frame.columns
    ]
    valid_features: list[str] = []

    for column in features:
        values = pd.to_numeric(
            frame[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        if not values.notna().any():
            continue
        if values.dropna().nunique() <= 1:
            continue
        valid_features.append(column)

    if "aqi" not in valid_features:
        raise ValueError(
            "The processed dataset must contain a non-constant current AQI column."
        )

    return valid_features


def _fit_scaler(
    values: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardization parameters using training rows only."""
    train_values = values[train_mask]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(train_values, axis=0)
        scale = np.nanstd(train_values, axis=0)

    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(
        np.isfinite(scale) & (scale > 1e-8),
        scale,
        1.0,
    )
    return mean.astype(np.float32), scale.astype(np.float32)


def _impute(
    values: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Impute missing values using medians from training rows only."""
    train_values = values[train_mask]
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(train_values, axis=0)

    medians = np.where(np.isfinite(medians), medians, 0.0)
    result = np.where(np.isfinite(values), values, medians)
    return result.astype(np.float32)


def _add_time_features_if_needed(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical features when the processed dataset does not already contain them."""
    result = frame.copy()
    if "datetime" not in result.columns:
        return result

    dt = pd.to_datetime(result["datetime"], errors="coerce")
    if "hour_sin" not in result.columns:
        hour = dt.dt.hour
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if "day_of_week_sin" not in result.columns:
        dow = dt.dt.dayofweek
        result["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        result["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    if "month_sin" not in result.columns:
        month = dt.dt.month
        result["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        result["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    if "is_weekend" not in result.columns:
        result["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)
    return result


def valid_sequence_endpoints(
    frame: pd.DataFrame,
    window: int,
    target_column: str = "target_aqi_next_hour",
) -> np.ndarray:
    """Return endpoints with continuous hourly history within each city."""
    if window < 2:
        raise ValueError("window must be at least 2.")
    if target_column not in frame.columns:
        raise ValueError(f"Required target column '{target_column}' is missing.")

    endpoints: list[np.ndarray] = []

    for _, group in frame.groupby("city_name", sort=False):
        group = group.sort_values("datetime")
        positions = group.index.to_numpy()
        times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        target = group[target_column].notna().to_numpy()

        if len(group) < window:
            continue

        hourly = (
            np.diff(times).astype("timedelta64[h]").astype(np.int64) == 1
        )
        consecutive = np.ones(len(group), dtype=bool)
        consecutive[1:] = hourly

        rolling_ok = np.ones(len(group), dtype=bool)
        for offset in range(1, window):
            rolling_ok[offset:] &= consecutive[offset:]

        eligible = positions[
            (np.arange(len(group)) >= window - 1)
            & rolling_ok
            & target
        ]
        if len(eligible):
            endpoints.append(eligible)

    if not endpoints:
        raise ValueError("No valid hourly sequences were found.")

    return np.concatenate(endpoints).astype(np.int64)


def build_sequence_data(
    frame: pd.DataFrame,
    window: int = 24,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[SequenceData, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare leakage-safe features and city-local chronological splits."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("val_fraction leaves no test period.")

    required = {"city_name", "datetime", "aqi", "target_aqi_next_hour"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Missing required sequence columns: "
            + ", ".join(sorted(missing))
        )

    frame = (
        frame.copy()
        .sort_values(["city_name", "datetime"])
        .reset_index(drop=True)
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")

    if frame["datetime"].isna().any():
        raise ValueError("The sequence dataset contains invalid datetime values.")
    if frame["city_name"].isna().any():
        raise ValueError("The sequence dataset contains missing city_name values.")

    if "carbon_dioxide" in frame.columns:
        frame["carbon_dioxide_missing"] = (
            pd.to_numeric(frame["carbon_dioxide"], errors="coerce")
            .isna()
            .astype(np.float32)
        )

    features = available_features(frame)

    co2_valid_fraction = (
        float(
            pd.to_numeric(
                frame["carbon_dioxide"], errors="coerce"
            ).notna().mean()
        )
        if "carbon_dioxide" in frame.columns
        else 0.0
    )

    if "carbon_dioxide" in frame.columns and co2_valid_fraction == 0.0:
        for column in ("carbon_dioxide", "carbon_dioxide_missing"):
            if column in features:
                features.remove(column)

    leakage = sorted({"target_aqi_next_hour"}.intersection(features))
    if leakage:
        raise ValueError(
            "Target leakage detected. Target columns cannot be model features: "
            + ", ".join(leakage)
        )

    values_frame = (
        frame[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    values = values_frame.to_numpy(dtype=np.float32)

    targets = (
        pd.to_numeric(
            frame["target_aqi_next_hour"], errors="coerce"
        )
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )
    current_aqi = (
        pd.to_numeric(frame["aqi"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )

    if not np.isfinite(targets).any():
        raise ValueError(
            "target_aqi_next_hour contains no valid numeric observations."
        )
    if not np.isfinite(current_aqi).any():
        raise ValueError("aqi contains no valid numeric observations.")

    timestamps = frame["datetime"].to_numpy()
    cities = frame["city_name"].astype(str).to_numpy()

    train_rows = np.zeros(len(frame), dtype=bool)
    train_idx_parts: list[np.ndarray] = []
    val_idx_parts: list[np.ndarray] = []
    test_idx_parts: list[np.ndarray] = []

    endpoints = valid_sequence_endpoints(frame, window)

    for city in pd.unique(cities):
        city_positions = np.flatnonzero(cities == city)
        city_times = timestamps[city_positions]

        city_start = city_times.min()
        city_end = city_times.max()
        duration_seconds = (
            city_end - city_start
        ).astype("timedelta64[s]").astype(np.int64)

        train_cut = city_start + np.timedelta64(
            int(duration_seconds * train_fraction),
            "s",
        )
        val_cut = city_start + np.timedelta64(
            int(duration_seconds * (train_fraction + val_fraction)),
            "s",
        )

        train_rows[city_positions] = city_times < train_cut

        city_endpoints = endpoints[cities[endpoints] == city]
        target_times = timestamps[city_endpoints] + np.timedelta64(1, "h")

        train_idx_parts.append(
            city_endpoints[target_times < train_cut]
        )
        val_idx_parts.append(
            city_endpoints[
                (target_times >= train_cut)
                & (target_times < val_cut)
            ]
        )
        test_idx_parts.append(
            city_endpoints[target_times >= val_cut]
        )

    if not train_rows.any():
        raise ValueError("No chronological training rows were found.")

    values = _impute(values, train_rows)
    mean, scale = _fit_scaler(values, train_rows)
    values = ((values - mean) / scale).astype(np.float32)

    train_idx = np.concatenate(
        [part for part in train_idx_parts if len(part)]
    ).astype(np.int64)
    val_idx = np.concatenate(
        [part for part in val_idx_parts if len(part)]
    ).astype(np.int64)
    test_idx = np.concatenate(
        [part for part in test_idx_parts if len(part)]
    ).astype(np.int64)

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("One chronological split has no valid sequences.")

    baseline = current_aqi.copy()
    train_median = np.nanmedian(baseline[train_rows])
    train_median = float(train_median) if np.isfinite(train_median) else 0.0
    baseline = np.where(
        np.isfinite(baseline),
        baseline,
        train_median,
    ).astype(np.float32)

    data = SequenceData(
        features=values,
        targets=targets,
        baseline_aqi=baseline,
        timestamps=timestamps,
        cities=cities,
        feature_columns=features,
        mean=mean,
        scale=scale,
        window=window,
        baseline_feature_index=features.index("aqi"),
    )
    return data, train_idx, val_idx, test_idx


class ResidualRegressor(nn.Module):
    """Base class for next-hour prediction around the persistence baseline."""

    def _init_residual_head(self, input_size: int) -> None:
        self.residual_head = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _predict_from_baseline(
        self,
        representation: torch.Tensor,
        baseline_aqi: torch.Tensor,
    ) -> torch.Tensor:
        correction = self.residual_head(representation).squeeze(-1)
        return baseline_aqi + correction


class LSTMRegressor(ResidualRegressor):
    """Two-layer LSTM residual regressor."""

    def __init__(self, input_size: int, hidden_size: int = 96, layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.20 if layers > 1 else 0.0,
        )
        self._init_residual_head(hidden_size)

    def forward(self, x: torch.Tensor, baseline_aqi: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self._predict_from_baseline(output[:, -1], baseline_aqi)


class GRURegressor(ResidualRegressor):
    """Two-layer GRU residual regressor."""

    def __init__(self, input_size: int, hidden_size: int = 96, layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.20 if layers > 1 else 0.0,
        )
        self._init_residual_head(hidden_size)

    def forward(self, x: torch.Tensor, baseline_aqi: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self._predict_from_baseline(output[:, -1], baseline_aqi)


class TCNBlock(nn.Module):
    """Residual dilated temporal convolution block."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.10):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=padding, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        if out.size(-1) != residual.size(-1):
            out = out[..., : residual.size(-1)]
        return self.activation(out + residual)


class TCNRegressor(ResidualRegressor):
    """Dilated residual TCN regressor."""

    def __init__(self, input_size: int, channels: int = 64):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(channels, 1),
            TCNBlock(channels, 2),
            TCNBlock(channels, 4),
            TCNBlock(channels, 8),
        )
        self._init_residual_head(channels)

    def forward(self, x: torch.Tensor, baseline_aqi: torch.Tensor) -> torch.Tensor:
        out = self.input_projection(x.transpose(1, 2))
        out = self.blocks(out)
        return self._predict_from_baseline(out[:, :, -1], baseline_aqi)


class PositionalEncoding(nn.Module):
    """Learned positional embeddings for Transformer sequences."""

    def __init__(self, window: int, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.size(1)]


class TransformerRegressor(ResidualRegressor):
    """Compact Transformer residual regressor."""

    def __init__(
        self,
        input_size: int,
        window: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.projection = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(window, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self._init_residual_head(d_model)

    def forward(self, x: torch.Tensor, baseline_aqi: torch.Tensor) -> torch.Tensor:
        out = self.projection(x)
        out = self.position(out)
        out = self.encoder(out)
        return self._predict_from_baseline(out[:, -1], baseline_aqi)


def create_model(name: str, input_size: int, window: int) -> nn.Module:
    """Create a supported residual forecasting model."""
    name = name.lower()
    if name == "lstm":
        return LSTMRegressor(input_size)
    if name == "gru":
        return GRURegressor(input_size)
    if name == "tcn":
        return TCNRegressor(input_size)
    if name == "transformer":
        return TransformerRegressor(input_size, window)
    raise ValueError(f"Unsupported model: {name}")


def subsample_indices(
    indices: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    """Sample endpoints reproducibly while preserving chronological ordering."""
    if max_samples is None or len(indices) <= max_samples:
        return np.asarray(indices, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(selected).astype(np.int64)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load a trained model and preprocessing metadata."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model = create_model(
        checkpoint["model_name"],
        len(checkpoint["feature_columns"]),
        int(checkpoint["window"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_sequence(
    model: nn.Module,
    sequence: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: str = "cpu",
    baseline_aqi: float | None = None,
) -> float:
    """Predict next-hour AQI from one unscaled sequence."""
    raw_sequence = sequence.astype(np.float32)
    if baseline_aqi is None:
        baseline_aqi = float(raw_sequence[-1, 0])

    scaled = ((raw_sequence - mean) / scale).astype(np.float32)
    tensor = torch.from_numpy(scaled).unsqueeze(0).to(device)
    baseline = torch.tensor([baseline_aqi], dtype=torch.float32, device=device)

    with torch.no_grad():
        return float(model(tensor, baseline).cpu().item())
