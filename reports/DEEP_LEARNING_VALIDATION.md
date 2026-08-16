# Deep-Learning Validation

## Changes validated

- Deep-learning models now predict a residual correction around the current-AQI persistence baseline.
- Train/validation/test splits are chronological **within each city**, so every city contributes to all three periods when enough history exists.
- CO2 handling is explicit:
  - `carbon_dioxide` is included when observed values exist.
  - `carbon_dioxide_missing` is included when CO2 exists but has missing observations.
  - Both are removed only when CO2 has no observed values at all.
- Missing numeric values are imputed from training-period medians.
- Scaling parameters are fitted from training rows only.
- Target leakage is rejected.
- Persistence remains an explicit benchmark in the deep-learning report.

## Pre-build model check

The model code was tested against the full raw Bangladesh AQI CSV after constructing the exact next-hour target and the fixed deep-learning feature set used by the project. The validation used 24-hour sequences, city-local 70/15/15 chronological splits, and 10,000/2,000/2,000 sampled train/validation/test sequences.

CO2 was partially observed in this dataset, so both CO2 and its missingness indicator were included.

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Persistence | 1.3952 | 3.9152 | 0.9938 |
| LSTM | 1.3784 | 3.6816 | 0.9946 |
| GRU | 1.3989 | 3.6857 | 0.9945 |
| TCN | **1.2433** | **3.0162** | **0.9963** |
| Transformer | 1.2744 | 3.3786 | 0.9954 |

These numbers are a pre-build engineering check, not a replacement for the final run on the project's generated `model_dataset.parquet`.

## Expected final command

After installing the project's requirements and generating `data/processed/model_dataset.parquet`, run:

```powershell
python src_train_deep_learning.py --input data\processed\model_dataset.parquet --models lstm gru tcn transformer --window 24 --epochs 20 --patience 4
```

The script reports the persistence baseline alongside every deep-learning model, prints CO2 inclusion status, and writes the checkpoints and metrics under `models/deep_learning/` and `reports/metrics/`.
