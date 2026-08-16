# src/06_train_deep_learning.py

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.deep_learning import create_model


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "processed" / "model_dataset.parquet"
DEFAULT_OUTPUT = ROOT / "models" / "deep_learning"
DEFAULT_REPORT = ROOT / "reports" / "metrics"

DEFAULT_WINDOW = 24
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train LSTM, GRU, TCN and Transformer AQI residual models."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


def detect_columns(frame: pd.DataFrame) -> tuple[str, str | None]:
    """Detect target and optional grouping columns."""
    if "aqi" not in frame.columns:
        raise ValueError(
            "model_dataset.parquet must contain an 'aqi' column."
        )

    group_column = None

    for candidate in (
        "city_name",
        "city",
        "location",
        "station",
        "station_name",
    ):
        if candidate in frame.columns:
            group_column = candidate
            break

    return "aqi", group_column


def prepare_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the dataset without leaking test information."""
    frame = frame.copy()

    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            errors="coerce",
        )
        frame = frame.dropna(
            subset=["datetime"]
        )

    target_column, group_column = detect_columns(frame)

    if group_column is None:
        frame["_group"] = "all"
        group_column = "_group"

    sort_columns = [group_column]

    if "datetime" in frame.columns:
        sort_columns.append("datetime")

    frame = frame.sort_values(
        sort_columns
    ).reset_index(drop=True)

    frame[target_column] = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[target_column]
    )

    if "carbon_dioxide" not in frame.columns:
        frame["carbon_dioxide"] = np.nan

    frame["carbon_dioxide_missing"] = (
        frame["carbon_dioxide"]
        .isna()
        .astype(np.float32)
    )

    return frame


def choose_features(
    frame: pd.DataFrame,
) -> list[str]:
    """Return the stable model feature schema."""
    preferred = [
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
        "is_weekend",
    ]

    features = [
        column
        for column in preferred
        if column in frame.columns
    ]

    if "aqi" not in features:
        raise ValueError(
            "AQI must be included as a model feature."
        )

    return features


def chronological_split(
    frame: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each location chronologically into train, validation and test."""
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in frame.groupby(
        group_column,
        sort=False,
    ):
        group = group.reset_index(drop=True)

        n = len(group)

        if n < 100:
            continue

        train_end = int(n * 0.70)
        validation_end = int(n * 0.85)

        train_parts.append(
            group.iloc[:train_end]
        )

        validation_parts.append(
            group.iloc[train_end:validation_end]
        )

        test_parts.append(
            group.iloc[validation_end:]
        )

    if not train_parts:
        raise ValueError(
            "No location contains enough observations for training."
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    test = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train, validation, test


def fit_scaler(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a feature scaler using training data only."""
    values = (
        train[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    values = values.ffill().bfill()

    means = values.mean(
        axis=0
    ).to_numpy(
        dtype=np.float32
    )

    scales = values.std(
        axis=0
    ).to_numpy(
        dtype=np.float32
    )

    means = np.nan_to_num(
        means,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scales = np.nan_to_num(
        scales,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    scales = np.where(
        scales > 1e-8,
        scales,
        1.0,
    ).astype(np.float32)

    return means, scales


def make_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    window: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build scaled feature windows, raw AQI baselines and raw targets."""
    x_values: list[np.ndarray] = []
    baselines: list[float] = []
    targets: list[float] = []

    for _, group in frame.groupby(
        group_column,
        sort=False,
    ):
        group = group.reset_index(drop=True)

        values = (
            group[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        values = values.ffill().bfill()

        if values.isna().any().any():
            continue

        raw = values.to_numpy(
            dtype=np.float32
        )

        if len(raw) <= window:
            continue

        scaled = (
            (raw - mean) / scale
        ).astype(np.float32)

        for end in range(
            window,
            len(raw),
        ):
            start = end - window

            sequence = scaled[
                start:end
            ]

            baseline = float(
                raw[end - 1, feature_columns.index("aqi")]
            )

            target = float(
                raw[end, feature_columns.index("aqi")]
            )

            x_values.append(sequence)
            baselines.append(baseline)
            targets.append(target)

    if not x_values:
        raise ValueError(
            "Unable to create any training sequences."
        )

    return (
        np.asarray(
            x_values,
            dtype=np.float32,
        ),
        np.asarray(
            baselines,
            dtype=np.float32,
        ),
        np.asarray(
            targets,
            dtype=np.float32,
        ),
    )


def make_loader(
    x: np.ndarray,
    baseline: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch data loader."""
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(baseline),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    error = predicted - actual

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )

    denominator = float(
        np.sum(
            (actual - actual.mean()) ** 2
        )
    )

    if denominator <= 1e-12:
        r2 = 0.0
    else:
        r2 = float(
            1.0
            - np.sum(error**2)
            / denominator
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training or validation epoch."""
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for x, baseline, y in loader:
        x = x.to(device)
        baseline = baseline.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            prediction = model(
                x,
                baseline,
            )

            loss = criterion(
                prediction,
                y,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = x.shape[0]

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_count += batch_size

    if total_count == 0:
        return float("inf")

    return total_loss / total_count


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate raw AQI predictions."""
    model.eval()

    predictions: list[np.ndarray] = []
    actual: list[np.ndarray] = []

    with torch.no_grad():
        for x, baseline, y in loader:
            x = x.to(device)
            baseline = baseline.to(device)

            output = model(
                x,
                baseline,
            )

            predictions.append(
                output.cpu().numpy()
            )

            actual.append(
                y.numpy()
            )

    return (
        np.concatenate(predictions),
        np.concatenate(actual),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    window: int,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Save a deployment-compatible checkpoint."""
    checkpoint: dict[str, Any] = {
        "model_name": model_name,
        "window": window,
        "feature_columns": feature_columns,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "architecture": "residual",
        "prediction_formula": (
            "baseline_aqi + learned_correction"
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def train_model(
    model_name: str,
    feature_columns: list[str],
    window: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    mean: np.ndarray,
    scale: np.ndarray,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate one residual model."""
    model = create_model(
        model_name,
        len(feature_columns),
        window,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    started = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        validation_loss = run_epoch(
            model,
            validation_loader,
            None,
            criterion,
            device,
        )

        scheduler.step(
            validation_loss
        )

        learning_rate_now = optimizer.param_groups[0]["lr"]

        print(
            f"[{model_name.upper()}] "
            f"epoch {epoch:02d}/{epochs} | "
            f"train={train_loss:.6f} | "
            f"val={validation_loss:.6f} | "
            f"lr={learning_rate_now:.2e}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(
                f"[{model_name.upper()}] "
                "early stopping."
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid checkpoint was produced for {model_name}."
        )

    model.load_state_dict(
        best_state
    )

    predicted, actual = predict_loader(
        model,
        test_loader,
        device,
    )

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    metrics["training_seconds"] = float(
        elapsed
    )

    metrics["parameters"] = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    save_checkpoint(
        output_dir / f"{model_name}.pt",
        model,
        model_name,
        window,
        feature_columns,
        mean,
        scale,
        metrics,
    )

    print(
        f"[{model_name.upper()}] "
        f"TEST MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f}"
    )

    return {
        "model": model_name,
        **metrics,
    }


def main() -> None:
    """Train all deep-learning models."""
    args = parse_args()

    if args.window < 2:
        raise ValueError(
            "--window must be at least 2."
        )

    set_seed(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading dataset: {args.input}"
    )

    frame = pd.read_parquet(
        args.input
    )

    frame = prepare_frame(
        frame
    )

    target_column, group_column = detect_columns(
        frame
    )

    assert group_column is not None

    feature_columns = choose_features(
        frame
    )

    print(
        f"Rows: {len(frame):,}"
    )

    print(
        f"Features: {feature_columns}"
    )

    print(
        f"Grouping column: {group_column}"
    )

    train, validation, test = chronological_split(
        frame,
        group_column,
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    mean, scale = fit_scaler(
        train,
        feature_columns,
    )

    train_x, train_baseline, train_y = make_sequences(
        train,
        feature_columns,
        group_column,
        args.window,
        mean,
        scale,
    )

    validation_x, validation_baseline, validation_y = make_sequences(
        validation,
        feature_columns,
        group_column,
        args.window,
        mean,
        scale,
    )

    test_x, test_baseline, test_y = make_sequences(
        test,
        feature_columns,
        group_column,
        args.window,
        mean,
        scale,
    )

    print(
        f"Train sequences: {len(train_x):,}"
    )

    print(
        f"Validation sequences: {len(validation_x):,}"
    )

    print(
        f"Test sequences: {len(test_x):,}"
    )

    train_loader = make_loader(
        train_x,
        train_baseline,
        train_y,
        args.batch_size,
        True,
    )

    validation_loader = make_loader(
        validation_x,
        validation_baseline,
        validation_y,
        args.batch_size,
        False,
    )

    test_loader = make_loader(
        test_x,
        test_baseline,
        test_y,
        args.batch_size,
        False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    results: list[dict[str, Any]] = []

    for model_name in (
        "lstm",
        "gru",
        "tcn",
        "transformer",
    ):
        result = train_model(
            model_name=model_name,
            feature_columns=feature_columns,
            window=args.window,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            mean=mean,
            scale=scale,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            device=device,
        )

        results.append(
            result
        )

    results_frame = pd.DataFrame(
        results
    ).sort_values(
        "rmse"
    )

    metrics_path = (
        args.report_dir
        / "deep_learning_metrics.csv"
    )

    results_frame.to_csv(
        metrics_path,
        index=False,
    )

    summary_path = (
        args.report_dir
        / "deep_learning_metrics.json"
    )

    summary = {
        "best_model": str(
            results_frame.iloc[0]["model"]
        ),
        "models": results,
        "feature_columns": feature_columns,
        "window": args.window,
        "device": str(device),
        "residual_architecture": True,
        "baseline": "raw_latest_aqi",
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FINAL DEEP-LEARNING TEST RESULTS")
    print("=" * 72)

    print(
        results_frame[
            [
                "model",
                "mae",
                "rmse",
                "r2",
                "training_seconds",
                "parameters",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Best deep-learning model: "
        f"{results_frame.iloc[0]['model']}"
    )

    print(
        f"Metrics written to: {metrics_path}"
    )

    print(
        f"JSON written to: {summary_path}"
    )

    print(
        f"Models written to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()