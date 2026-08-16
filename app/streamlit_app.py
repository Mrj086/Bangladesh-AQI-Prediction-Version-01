"""Streamlit dashboard for Bangladesh AQI forecasting research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models" / "deep_learning"
PROCESSED = ROOT / "data" / "processed" / "model_dataset.parquet"

st.set_page_config(
    page_title="Bangladesh AQI Intelligence",
    page_icon="🌏",
    layout="wide",
)

st.title("Bangladesh AQI Forecasting Intelligence")
st.caption(
    "Research-grade next-hour forecasting | Classical ML + Deep Learning"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Classical ML",
        "Deep Learning",
        "City Forecast",
        "Research Artifacts",
    ],
)


def load_json(path: Path) -> dict:
    """Load a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: object) -> str:
    """Format numeric dashboard values."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


if page == "Overview":
    st.header("Project overview")

    final_path = METRICS / "final_metrics.json"
    dl_path = METRICS / "deep_learning_results.csv"

    if final_path.exists():
        final = load_json(final_path)
        metric = final.get("final_test_metrics", {})
        cols = st.columns(4)
        cols[0].metric("Best ML model", final.get("best_model", "N/A"))
        cols[1].metric("ML MAE", format_metric(metric.get("mae", float("nan"))))
        cols[2].metric("ML RMSE", format_metric(metric.get("rmse", float("nan"))))
        cols[3].metric("ML R²", format_metric(metric.get("r2", float("nan"))))
    else:
        st.info("Classical ML artifacts are not available yet.")

    if dl_path.exists():
        dl = pd.read_csv(dl_path).sort_values("rmse")
        best_dl = dl.iloc[0]
        st.subheader("Deep-learning benchmark")
        cols = st.columns(4)
        cols[0].metric("Best DL model", str(best_dl["model"]))
        cols[1].metric("DL MAE", f"{best_dl['mae']:.4f}")
        cols[2].metric("DL RMSE", f"{best_dl['rmse']:.4f}")
        cols[3].metric("DL R²", f"{best_dl['r2']:.4f}")
        st.dataframe(dl, use_container_width=True)
    else:
        st.info(
            "Run src/06_train_deep_learning.py to populate the deep-learning benchmark."
        )

    st.markdown(
        """
### Research pipeline

**Audit → Cleaning → Outlier treatment → Feature engineering → Time-aware
validation → Classical ML → LSTM/GRU/TCN/Transformer → Ensemble →
Error analysis → Research reporting**
"""
    )

elif page == "Classical ML":
    st.header("Classical ML benchmark")

    cv_path = METRICS / "cross_validation_results.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path).sort_values("cv_mean_rmse")
        st.dataframe(cv, use_container_width=True)
        st.subheader("Cross-validation RMSE")
        st.bar_chart(cv.set_index("model")["cv_mean_rmse"])
    else:
        st.warning("Run the classical training pipeline first.")

    for image_name in [
        "actual_vs_predicted.png",
        "residual_distribution.png",
        "city_rmse.png",
    ]:
        path = FIGURES / image_name
        if path.exists():
            st.image(str(path), caption=image_name, use_container_width=True)

elif page == "Deep Learning":
    st.header("Deep-learning benchmark")

    results_path = METRICS / "deep_learning_results.csv"
    predictions_path = METRICS / "deep_learning_predictions.parquet"

    if not results_path.exists():
        st.warning(
            "Deep-learning artifacts are missing. Run "
            "`python src\\06_train_deep_learning.py --input "
            "\"data\\processed\\model_dataset.parquet\"`."
        )
        st.stop()

    results = pd.read_csv(results_path).sort_values("rmse")
    st.dataframe(results, use_container_width=True)

    st.subheader("RMSE comparison")
    st.bar_chart(results.set_index("model")["rmse"])

    for image_name in [
        "dl_model_rmse_comparison.png",
        "dl_lstm_training_curve.png",
        "dl_gru_training_curve.png",
        "dl_tcn_training_curve.png",
        "dl_transformer_training_curve.png",
        "dl_ensemble_recent_forecast.png",
    ]:
        image_path = FIGURES / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name, use_container_width=True)

    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        cities = sorted(predictions["city_name"].unique())
        selected_city = st.selectbox("City", cities)
        city_data = predictions[
            predictions["city_name"] == selected_city
        ].sort_values("datetime").tail(500)

        prediction_columns = [
            c for c in city_data.columns if c.endswith("_prediction")
        ]
        st.line_chart(
            city_data.set_index("datetime")[
                ["actual_aqi"] + prediction_columns
            ]
        )

        st.download_button(
            "Download deep-learning predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="deep_learning_predictions.csv",
            mime="text/csv",
        )

elif page == "City Forecast":
    st.header("Live deep-learning next-hour forecast")

    if not PROCESSED.exists():
        st.warning("Processed model dataset not found.")
        st.stop()

    metadata_path = METRICS / "deep_learning_metadata.json"
    results_path = METRICS / "deep_learning_results.csv"

    if not metadata_path.exists() or not results_path.exists():
        st.warning("Train the deep-learning benchmark first.")
        st.stop()

    try:
        import torch

        from src.deep_learning import load_checkpoint, predict_sequence
    except ImportError as exc:
        st.error(f"PyTorch inference dependencies are unavailable: {exc}")
        st.stop()

    metadata = load_json(metadata_path)
    results = pd.read_csv(results_path)

    checkpoint_models = [
        model_name
        for model_name in results["model"].tolist()
        if model_name != "weighted_ensemble"
        and (MODELS / f"{model_name}.pt").exists()
    ]

    if not checkpoint_models:
        st.warning("No trained deep-learning checkpoint is available.")
        st.stop()

    default_model = metadata.get("best_model", checkpoint_models[0])

    if default_model not in checkpoint_models:
        default_model = checkpoint_models[0]

    model_name = st.selectbox(
        "Model",
        checkpoint_models,
        index=checkpoint_models.index(default_model),
    )

    frame = pd.read_parquet(PROCESSED)

    cities = sorted(
        frame["city_name"]
        .dropna()
        .unique()
    )

    city = st.selectbox("City", cities)

    window = int(metadata["window_hours"])
    feature_columns = list(metadata["feature_columns"])

    city_frame = (
        frame[frame["city_name"] == city]
        .sort_values("datetime")
        .tail(window)
        .copy()
    )

    if len(city_frame) < window:
        st.error(
            f"Not enough observations for {city}. "
            f"Required: {window}, available: {len(city_frame)}."
        )
        st.stop()

    missing = [
        column
        for column in feature_columns
        if column not in city_frame.columns
        and column != "carbon_dioxide_missing"
    ]

    if missing:
        st.error(f"Missing inference features: {missing}")
        st.stop()

    if "carbon_dioxide_missing" in feature_columns:
        if "carbon_dioxide" not in city_frame.columns:
            st.error(
                "The model requires carbon_dioxide, but the dataset "
                "does not contain that column."
            )
            st.stop()

        city_frame["carbon_dioxide_missing"] = (
            city_frame["carbon_dioxide"]
            .isna()
            .astype("float32")
        )

    if "carbon_dioxide" in feature_columns:
        city_frame["carbon_dioxide"] = (
            pd.to_numeric(
                city_frame["carbon_dioxide"],
                errors="coerce",
            )
            .ffill()
            .bfill()
            .fillna(0.0)
        )

    checkpoint_path = MODELS / f"{model_name}.pt"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    try:
        model, checkpoint = load_checkpoint(
            checkpoint_path,
            device=device,
        )
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

    sequence_frame = (
        city_frame[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [float("inf"), -float("inf")],
            pd.NA,
        )
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    sequence = sequence_frame.to_numpy(
        dtype="float32",
    )

    try:
        prediction = predict_sequence(
            model,
            sequence,
            checkpoint["mean"],
            checkpoint["scale"],
            device=device,
        )
    except Exception as exc:
        st.error(
            f"Model inference failed: {exc}"
        )
        st.stop()

    last_time = city_frame["datetime"].iloc[-1]

    forecast_time = (
        last_time
        + pd.Timedelta(hours=1)
    )

    cols = st.columns(3)

    cols[0].metric(
        "City",
        city,
    )

    cols[1].metric(
        "Forecast time",
        forecast_time.strftime(
            "%Y-%m-%d %H:%M"
        ),
    )

    cols[2].metric(
        "Predicted AQI",
        f"{prediction:.2f}",
    )

    st.subheader("Input history")

    display_columns = [
        "datetime",
        "aqi",
    ] + [
        column
        for column in feature_columns
        if column != "aqi"
    ]

    st.dataframe(
        city_frame[display_columns].tail(window),
        use_container_width=True,
    )

elif page == "Research Artifacts":
    st.header("Research artifacts")

    files = sorted(
        [
            p
            for directory in [METRICS, FIGURES, MODELS]
            if directory.exists()
            for p in directory.rglob("*")
            if p.is_file()
        ]
    )

    if not files:
        st.info("No artifacts have been generated yet.")
    else:
        for path in files:
            st.write(str(path.relative_to(ROOT)))
