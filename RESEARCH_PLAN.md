# Research Plan

## Proposed contribution

A reproducible, leakage-safe benchmark for next-hour AQI forecasting across Bangladeshi cities, comparing classical machine learning with recurrent, convolutional, and attention-based deep learning.

The project emphasizes:

- strict temporal evaluation
- train-only preprocessing
- city-aware sequence construction
- missing-data robustness
- extreme-event error analysis
- explainability
- reproducibility
- deployability

## Main research question

> Can historical AQI, pollutant dynamics, temporal context, and spatial information improve next-hour AQI forecasting across heterogeneous Bangladeshi cities, and which model family provides the best accuracy-to-complexity trade-off?

## Hypotheses

H1. Historical AQI and pollutant dynamics provide useful signal for next-hour AQI.

H2. Nonlinear ensemble models outperform a regularized linear baseline.

H3. Historical lag and rolling information improves forecasting over contemporaneous measurements alone.

H4. Deep sequence models can capture temporal dependencies that are not fully represented by tabular features.

H5. TCN and Transformer architectures can provide competitive performance against recurrent models.

H6. A validation-weighted ensemble improves robustness over individual deep-learning models.

H7. Forecast performance varies across cities, seasons, and pollution severity.

H8. Extreme-pollution periods have larger absolute forecast errors than normal periods.

## Experiments

### E1 — Persistence baseline

```text
AQI(t+1) = AQI(t)
```

### E2 — Linear ML

Ridge with leakage-safe preprocessing.

### E3 — Bagging ML

- Random Forest
- Extra Trees

### E4 — Boosting ML

- Gradient Boosting
- HistGradientBoosting

### E5 — Classical time-series validation

Rolling-origin cross-validation using unique timestamps.

### E6 — Deep recurrent models

- LSTM
- GRU

### E7 — Deep convolutional model

Temporal Convolutional Network with residual dilated blocks.

### E8 — Attention model

Compact Transformer Encoder over a 24-hour multivariate sequence.

### E9 — Deep ensemble

Validation-weighted ensemble using inverse validation RMSE.

The final test set must not be used to determine ensemble weights.

### E10 — Feature ablation

Compare:

1. current measurements
2. + temporal features
3. + historical lags
4. + rolling statistics
5. + spatial/city information

### E11 — Horizon analysis

Extend the framework to:

- 1-hour
- 6-hour
- 12-hour
- 24-hour

forecast horizons.

### E12 — City generalization

Train on selected cities and test on cities excluded from training.

### E13 — Distribution shift

Compare performance across:

- years
- seasons
- hours
- cities
- pollution severity

### E14 — Explainability

Use:

- permutation importance
- tree-based feature importance
- SHAP where appropriate
- deep-learning attribution methods for selected experiments

### E15 — Robustness

Repeat the final benchmark over multiple random seeds and report variability.

### E16 — Statistical comparison

For paired test forecasts, consider:

- Diebold-Mariano testing
- bootstrap confidence intervals
- paired absolute-error comparisons

Use these only after the experimental protocol has been frozen.

## Evaluation metrics

Primary:

- MAE
- RMSE
- R²

Secondary:

- MAPE
- median absolute error
- explained variance

Operational:

- error during high-AQI events
- city-level error
- seasonal error
- computational cost
- parameter count

## Paper tables

1. Dataset summary
2. Missingness summary
3. Feature groups
4. Classical model hyperparameters
5. Deep-learning architecture/configuration
6. Classical CV metrics
7. Final ML test metrics
8. Deep-learning validation/test metrics
9. ML vs DL comparison
10. City-level metrics
11. Seasonal metrics
12. High-AQI episode metrics
13. Ablation results
14. Feature importance
15. Computational cost

## Paper figures

1. End-to-end study workflow
2. Bangladesh city coverage
3. AQI temporal trend
4. Missingness profile
5. AQI/pollutant distributions
6. Correlation matrix
7. Hour × month heatmap
8. Classical CV comparison
9. Deep-learning training curves
10. ML vs DL benchmark
11. Actual vs predicted AQI
12. Residual distribution
13. City-level RMSE
14. High-AQI error analysis
15. Recent 24-hour forecast example

## Publication discipline

Do not choose the best model based on the final test set.

Do not calculate preprocessing statistics using future observations.

Do not let a sequence cross a city boundary.

Do not allow a sequence to cross a missing-hour gap.

Do not report only the best-performing city.

Do not hide poor-performing cases.

Do not claim causal relationships from predictive feature importance.

Report exact data provenance, source, access date, AQI methodology, and license.

If redistribution is restricted, distribute code and reproducible download instructions rather than the raw dataset.

## Reproducibility

Record:

- dataset version/hash
- source URL
- access date
- Python version
- package versions
- operating system
- random seed
- model configuration
- training sample counts
- time split boundaries
- test period
- hardware
- training time

