# Daily Max Temperature Neural Network

A bias-correction neural network that predicts daily maximum temperatures for 20 US cities by learning the residual between NWS recorded highs and numerical weather prediction (NWP) forecast ensemble means.

## How It Works

Instead of predicting temperature from scratch, the model corrects systematic biases in existing NWP forecasts:

```
Predicted Actual Temperature = NWP Forecast Ensemble Mean + Predicted Residual
```

The model outputs a heteroscedastic Gaussian (mu, sigma), providing both a point prediction and calibrated uncertainty estimate.

## Architecture

- **Type**: MLP with city embeddings
- **Input**: 27 continuous features + 8-dim city embedding
- **Hidden layers**: [128, 64, 32] with BatchNorm, GELU, dropout
- **Output**: (mu, sigma) — heteroscedastic Gaussian
- **Loss**: Gaussian negative log-likelihood
- **Parameters**: ~15,000

## Features (27 continuous)

| Category | Features |
|---|---|
| Forecast ensemble stats | mean, std, range, IQR across 5 NWP models |
| Individual NWP outputs | GFS, ECMWF, ICON, GEM, JMA |
| Pairwise spreads | ECMWF-GFS, ICON-GFS, GEM-GFS, JMA-GFS |
| Lagged meteorology | cloud cover, dewpoint, wind, pressure, precip (yesterday) |
| Lagged hourly temps | 6am, 9am, noon, 3pm, diurnal range (yesterday) |
| Rolling forecast bias | 7-day, 14-day, 30-day rolling mean residual |
| City static | lat, lon, elevation, coastal, continentality |
| Calendar | sin/cos day-of-year, sin/cos month |

## Performance (Out-of-Sample Test: Jan–Apr 2026)

| Metric | Value |
|---|---|
| MAE | 1.11°F |
| RMSE | 1.49°F |
| Bias | -0.02°F |
| R² | 0.994 |
| Correlation | 0.997 |

The model beats all 5 individual NWP models and their ensemble mean on the test set.

## Data

All data is included in the `data/` directory:

- `data/nws_daily/` — NWS daily recorded highs from ACIS (ground truth target)
- `data/weather_forecasts/` — Historical NWP model forecasts from Open-Meteo
- `data/weather_archive/` — Historical daily + hourly weather from Open-Meteo (features)
- `data/climate_indices/` — ENSO, AO, NAO, PNA from NOAA CPC

**20 cities**: New York, Chicago, Miami, Boston, Los Angeles, Austin, San Francisco, Dallas, Philadelphia, Phoenix, Oklahoma City, Denver, Washington DC, San Antonio, Houston, Minneapolis, Atlanta, Seattle, Las Vegas, New Orleans

**Date range**: 2022-01-01 to 2026-04-16

## Usage

### Train the model

```bash
pip install -r requirements.txt
python model1_forecast.py
```

This builds features, trains the model with early stopping, and saves:
- `checkpoints/model1_best.pt` — best model weights
- `checkpoints/model1_scaler.pkl` — fitted StandardScaler
- `checkpoints/model1_preds_val.csv` — validation predictions
- `checkpoints/model1_preds_test.csv` — test predictions

### Compare against NWP forecasts

```bash
python compare_models.py
```

Prints MAE/RMSE/bias tables comparing the neural net against GFS, ECMWF, ICON, GEM, JMA, and their ensemble mean across train/val/test splits.

### Out-of-sample analysis notebook

```bash
jupyter notebook model1_oos.ipynb
```

Detailed evaluation: per-city metrics, calibration plots, residual diagnostics, time series visualization, MAE heatmaps.

### Refresh data (optional)

```bash
python data_fetch.py
```

Re-fetches all data from APIs (NWS/ACIS, Open-Meteo, NOAA CPC). Existing files are skipped.

## Data Splits

| Split | Date Range | Purpose |
|---|---|---|
| Train | 2022-01-01 to 2024-12-31 | Model training |
| Validation | 2025-01-01 to 2025-12-31 | Early stopping / hyperparameter tuning |
| Test | 2026-01-01 to 2026-04-16 | Out-of-sample evaluation |
