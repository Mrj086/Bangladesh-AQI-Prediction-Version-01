# Bangladesh AQI Forecasting & Urban Pollution Intelligence

Research-grade, leakage-safe **next-hour AQI forecasting** project for Bangladesh, extended with a reproducible deep-learning benchmark and Streamlit deployment.

## TL;DR

The project now contains two complementary modeling tracks:

**Classical ML**
- Ridge
- Random Forest
- Extra Trees
- HistGradientBoosting
- Gradient Boosting
- Time-aware rolling cross-validation

**Deep Learning**
- LSTM
- GRU
- Temporal Convolutional Network (TCN)
- Transformer Encoder
- validation-based weighted ensemble

The prediction task is:

> Given information available at time `t`, forecast AQI at `t + 1 hour` for the same city.

The project explicitly avoids random train/test splitting and prevents preprocessing leakage.

---

## 1. Project structure

```text
aqi-bangladesh-forecasting/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── AQI Bangladesh.csv
│   └── processed/
├── models/
│   └── deep_learning/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── 01_data_audit.py
│   ├── 02_build_features.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   ├── 05_eda.py
│   ├── 06_train_deep_learning.py
│   ├── common.py
│   ├── deep_learning.py
│   └── modeling.py
├── tests/
├── config.yaml
├── requirements.txt
├── RESEARCH_PLAN.md
├── Dockerfile
└── README.md
```

---

## 2. Dataset

The supplied public dataset contains approximately:

- **1.05 million rows**
- **30 cities**
- hourly observations
- time coverage from **2000-01-01 to 2025-11-23**
- pollutant measurements and AQI

The supplied schema includes:

```text
city_id
city_name
lat
lon
datetime
pm10
pm2_5
carbon_monoxide
carbon_dioxide
nitrogen_dioxide
sulphur_dioxide
ozone
aqi
```

`carbon_dioxide` is substantially missing in the supplied data. The pipeline detects completely empty model features and excludes them instead of attempting impossible median imputation.

**Research requirement:** record the original dataset URL, provider, access date, license, and methodology before publishing the paper. Do not claim ownership of the public data.

---

# 3. Environment setup — Windows

Open Anaconda Prompt.

```bat
cd "C:\Users\USER\Downloads\aqi-bangladesh-forecasting\aqi-bangladesh-forecasting"

conda activate aqi-research

python --version

pip install -r requirements.txt
```

Verify PyTorch:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Verify Streamlit:

```bat
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

---

# 4. Put the data in the correct location

The raw CSV should be:

```text
data\raw\AQI Bangladesh.csv
```

Check:

```bat
dir "data\raw"
```

---

# 5. Run the complete classical ML pipeline

## Step 1 — Data audit

```bat
python src\01_data_audit.py --input "data\raw\AQI Bangladesh.csv"
```

This creates the audit report under:

```text
reports\metrics\
```

---

## Step 2 — Feature engineering

```bat
python src\02_build_features.py --input "data\raw\AQI Bangladesh.csv"
```

Expected output:

```text
data\processed\model_dataset.parquet
```

This stage performs:

- datetime normalization
- duplicate removal
- physical plausibility checks
- temporal features
- cyclical encoding
- lag features
- rolling statistics
- first differences
- next-hour target construction

---

## Step 3 — Classical ML

```bat
python src\03_train.py --input "data\processed\model_dataset.parquet"
```

For a faster development run:

```bat
python src\03_train.py --input "data\processed\model_dataset.parquet" --sample-fraction 0.20
```

**Use the full dataset for the final research run.**

The training pipeline uses chronological validation and a final chronological test period.

---

## Step 4 — Evaluation

```bat
python src\04_evaluate.py --input "data\processed\model_dataset.parquet"
```

Evaluation artifacts are saved under:

```text
reports\figures\
reports\metrics\
```

---

# 6. Deep-learning pipeline

This is the new research branch.

## Models

### 1. LSTM

Good at learning sequential dependencies and long-term temporal patterns.

### 2. GRU

A simpler recurrent architecture that can approach LSTM performance with fewer parameters.

### 3. TCN

Uses dilated causal-style temporal convolutions and residual blocks.

### 4. Transformer Encoder

Uses attention to learn relationships between different positions in the historical window.

The benchmark also produces a **validation-weighted ensemble**.

The ensemble weights are calculated from validation performance, not test performance. This prevents test-set leakage.

---

# 7. Deep-learning training

Run:

```bat
python src\06_train_deep_learning.py --input "data\processed\model_dataset.parquet"
```

Default settings are intentionally practical for a large dataset:

```text
window = 24 hours
train sequences = up to 150,000
validation sequences = up to 30,000
test sequences = up to 30,000
batch size = 256
maximum epochs = 20
early stopping patience = 4
```

The 24-hour window means:

```text
t-23
t-22
...
t-2
t-1
t
 ↓
AQI(t+1)
```

### Run more training data

For a stronger final experiment:

```bat
python src\06_train_deep_learning.py ^
  --input "data\processed\model_dataset.parquet" ^
  --max-train-sequences 300000 ^
  --max-val-sequences 50000 ^
  --max-test-sequences 50000 ^
  --epochs 30
```

### Run one model first

For a quick test:

```bat
python src\06_train_deep_learning.py ^
  --input "data\processed\model_dataset.parquet" ^
  --models lstm ^
  --max-train-sequences 10000 ^
  --max-val-sequences 2000 ^
  --max-test-sequences 2000 ^
  --epochs 3
```

Then expand to all four models.

---

# 8. Deep-learning outputs

After successful training:

```text
models/
└── deep_learning/
    ├── lstm.pt
    ├── gru.pt
    ├── tcn.pt
    └── transformer.pt
```

and:

```text
reports/
└── metrics/
    ├── deep_learning_results.csv
    ├── deep_learning_predictions.parquet
    ├── deep_learning_metadata.json
    └── deep_learning_history.json
```

The results table contains:

- MAE
- RMSE
- R²
- MAPE
- validation RMSE
- training time
- parameter count

---

# 9. Why the deep-learning branch is leakage-safe

The pipeline does not randomly split sequences.

It first defines chronological train/validation/test periods.

It then:

1. Fits feature imputation using training rows.
2. Fits scaling using training rows only.
3. Creates hourly sequences inside each city.
4. Rejects sequences crossing missing-hour gaps.
5. Keeps city histories separate.
6. Assigns a sequence to a split according to the time of its forecast target.
7. Keeps the final test period untouched.

This is important for publication-quality forecasting research.

---

# 10. Outlier handling

Outliers are not blindly deleted.

For classical ML, numeric preprocessing uses:

```text
median imputation
      ↓
1st–99th percentile clipping
      ↓
standardization
```

The thresholds are learned from the training fold.

For deep learning, the input sequence is standardized using training-only statistics. Extreme observations are retained rather than silently deleted.

This distinction should be described in the paper.

---

# 11. Missing carbon dioxide

The supplied dataset has extensive `carbon_dioxide` missingness.

If a feature is completely empty:

```text
carbon_dioxide
carbon_dioxide_lag_1h
...
```

the classical pipeline excludes those columns before imputation.

This prevents the scikit-learn warning:

```text
Skipping features without any observed values
```

If a feature has some valid observations, it can still be retained and imputed using training-only statistics.

---

# 12. Cross-validation

Classical ML uses rolling-origin time-series cross-validation.

The folds are created using unique timestamps rather than random rows.

Conceptually:

```text
Fold 1
TRAIN TRAIN TRAIN | VALID

Fold 2
TRAIN TRAIN TRAIN VALID | VALID

Fold 3
TRAIN TRAIN TRAIN VALID VALID | VALID

Fold 4
TRAIN TRAIN TRAIN VALID VALID VALID | VALID
```

The final test period is kept separate.

Deep learning uses a chronological train/validation/test protocol because full multi-model deep-learning cross-validation on more than one million observations is computationally expensive.

For the research paper, this distinction should be explicitly reported rather than pretending all models used identical validation procedures.

---

# 13. Streamlit dashboard

After generating artifacts:

```bat
streamlit run app\streamlit_app.py
```

Open the displayed local URL, normally:

```text
http://localhost:8501
```

The dashboard contains:

### Overview

- best classical model
- best deep-learning model
- MAE
- RMSE
- R²
- benchmark table

### Classical ML

- CV results
- RMSE comparison
- actual-vs-predicted plot
- residual plot
- city RMSE

### Deep Learning

- LSTM/GRU/TCN/Transformer comparison
- ensemble
- city-level prediction curves
- downloadable prediction data

### City Forecast

Select:

```text
City
Model
```

The application loads the trained checkpoint and predicts the next-hour AQI using the latest 24-hour history.

### Research Artifacts

Browse generated metrics, figures, and model artifacts.

---

# 14. FastAPI

The existing classical-model API can be started with:

```bat
uvicorn app.api:app --reload
```

Health check:

```text
GET /health
```

The API is intentionally separate from the Streamlit research dashboard.

---

# 15. Tests

Run:

```bat
pytest -q
```

The test suite covers:

- metric calculation
- zero-safe MAPE
- next-hour target construction
- chronological sequence splitting
- all deep-learning model tensor shapes

Before packaging, the project was syntax-checked and the automated test suite passed.

---

# 16. Research-paper experiment matrix

Your final paper should compare at least:

| Model | Type | CV | Test MAE | Test RMSE | Test R² | Time |
|---|---|---|---:|---:|---:|---:|
| Persistence | Baseline | — | | | | |
| Ridge | ML | ✓ | | | | |
| Random Forest | ML | ✓ | | | | |
| Extra Trees | ML | ✓ | | | | |
| HistGradientBoosting | ML | ✓ | | | | |
| Gradient Boosting | ML | ✓ | | | | |
| LSTM | DL | temporal validation | | | | |
| GRU | DL | temporal validation | | | | |
| TCN | DL | temporal validation | | | | |
| Transformer | DL | temporal validation | | | | |
| Weighted Ensemble | DL ensemble | temporal validation | | | | |

Do not select the final model based on test RMSE.

Select the model using development/validation evidence, then report its untouched test performance.

---

# 17. Strong research questions

### RQ1

Can historical pollutant concentrations and temporal dynamics accurately forecast next-hour AQI?

### RQ2

Do recurrent deep-learning models outperform classical tree-based models?

### RQ3

Does TCN or Transformer attention provide an advantage over recurrent architectures?

### RQ4

How does forecast accuracy change across cities?

### RQ5

How does performance change during extreme pollution events?

### RQ6

Which historical pollutant and temporal signals are most predictive?

### RQ7

Does an ensemble improve robustness over individual models?

---

# 18. Recommended paper figures

Use these as the main paper figures:

1. Dataset coverage and city distribution
2. AQI distribution
3. Missingness profile
4. Seasonal/hourly AQI heatmap
5. Pollutant correlation matrix
6. Feature importance
7. CV model comparison
8. Deep-learning training curves
9. ML vs DL test performance
10. Actual vs predicted AQI
11. Residual distribution
12. City-level RMSE
13. High-AQI error analysis
14. Example 24-hour forecasts

---

# 19. Important scientific caution

A higher R² does not automatically mean a model is operationally better.

Report:

- MAE
- RMSE
- R²
- high-AQI performance
- city-level stability
- computational cost

Also compare against the persistence baseline.

Most importantly:

> Predictive importance is not causal importance.

Do not write that PM2.5 "causes" a model prediction to increase unless a separate causal study supports that conclusion.

---

# 20. Data licensing and research use

The fact that a dataset is publicly downloadable does **not** automatically mean it has no copyright or licensing restrictions.

Before publishing:

1. Find the original source.
2. Read its license/terms.
3. Cite the dataset.
4. Cite the original data provider.
5. State the access date.
6. Check whether redistribution is allowed.
7. If redistribution is restricted, distribute your code and provide instructions for obtaining the data instead of including the raw CSV.

Do not claim the public dataset as your own collection.

---

# 21. Recommended final workflow

```text
1. Data provenance
       ↓
2. Audit
       ↓
3. Cleaning
       ↓
4. Feature engineering
       ↓
5. Leakage audit
       ↓
6. Classical ML + temporal CV
       ↓
7. Deep learning benchmark
       ↓
8. Validation-based ensemble
       ↓
9. Final untouched test
       ↓
10. City/season/extreme-event analysis
       ↓
11. Explainability
       ↓
12. Statistical comparison
       ↓
13. Streamlit deployment
       ↓
14. Research paper
```

---

# 22. Final quality checklist

Before calling the project "final":

- [ ] Original data source documented
- [ ] Dataset license documented
- [ ] No duplicate city-timestamps
- [ ] Negative physical values handled
- [ ] Completely empty features removed
- [ ] Missingness documented
- [ ] Outlier strategy documented
- [ ] No future features
- [ ] Training-only preprocessing
- [ ] Persistence baseline
- [ ] At least 4 classical ML models
- [ ] LSTM
- [ ] GRU
- [ ] TCN
- [ ] Transformer
- [ ] Time-aware CV for classical models
- [ ] Chronological DL validation
- [ ] Final untouched test period
- [ ] City-level metrics
- [ ] Extreme-event analysis
- [ ] Explainability
- [ ] Reproducible random seed
- [ ] Saved checkpoints
- [ ] Streamlit dashboard
- [ ] Unit tests
- [ ] README
- [ ] Research plan
- [ ] Paper tables/figures
- [ ] Limitations section
