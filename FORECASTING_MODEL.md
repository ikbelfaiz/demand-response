# Household forecasting model

## Reused notebook model

Both `energy_ttm_d1_forecasting.ipynb` and `energy_ttm_dr_backtesting_2025.ipynb` unambiguously use the final `EnergyFM/energy-ttm` checkpoint through `TinyTimeMixerForPrediction`. They use one input channel, context-only mean/std normalization, zero-shot inference, and clipping of physically impossible negative power. They do not train a household model or save a model artifact. The larger notebook backtests all 358 eligible 2025 days at hourly zone level: Energy-TTM MAE 33.050 MW versus 31.056 MW for 24-hour persistence, so it honestly reports a -6.42% improvement.

The notebooks' Random Forest is a separate DR-event classifier, not a load forecaster, and was not substituted for Energy-TTM.

## Half-hour adaptation

The checkpoint expects 168 hourly values and predicts 24 hourly values. The application objective covers the same durations at twice the temporal resolution. Its patch length and stride are therefore changed from 24 to 48 while preserving seven daily patches; context length changes from 168 to 336 and the forecast head from 24 to 48. A dimension check showed that changing only context and horizon is invalid. The mismatched input patcher and forecast head are reinitialized by Granite-TSFM, then all 30,610 parameters are fine-tuned while retaining the pretrained mixer core.

The single target is household aggregate power in kW. Each predicted interval's energy is explicitly `power kW × 0.5 h`; totals are never calculated by summing kW as though it were energy. No uncertainty band is shown because this implementation does not calibrate one.

## Data and leakage controls

- Canonical sources: both `data/households_1min_*.parquet` files.
- Operational input: mean power at 30-minute resolution through the existing aggregation service.
- One global channel-independent model is trained across household-labelled windows; 50 separate models are not created.
- Train target days: 2025-01-08 through 2025-09-30.
- Validation target days: 2025-10-01 through 2025-10-31.
- Test target days: 2025-11-01 through 2025-12-31.
- Windows are split by target day before batching. Evaluation targets never overlap training targets.
- Each origin is normalized only with the 336 already-observed context values, matching the notebook. There is no dataset-wide fitted scaler to leak future values or serialize.
- Missing context slots are rejected, never interpolated by the forecast pipeline. The canonical installed aggregate sensor is currently complete.

## Training, evaluation, and artifacts

Use Python 3.13 because Granite-TSFM does not support Python 3.14:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 scripts/train_forecasting.py --households 50 --epochs 5 --batch-size 64
py -3.13 scripts/evaluate_forecasting.py --households 50
```

The artifact directory is `models/household_forecasting/energy_ttm_v1/`. It contains the saved model, versioned metadata, training history, overall results, per-household results, per-month results, and window-level metrics. Evaluation compares Energy-TTM with seasonal persistence (the same half-hour slots seven days earlier) using MAE, RMSE, WAPE, daily energy error, peak magnitude error, and peak timing error.

The executed test covers 3,050 windows (50 households × 61 days, November 1–December 31, 2025). Mean window metrics were:

| Method | MAE (kW) | RMSE (kW) | WAPE | Daily energy error (kWh) | Peak magnitude error (kW) | Peak timing error (minutes) |
|---|---:|---:|---:|---:|---:|---:|
| Energy-TTM | 0.2472 | 0.3865 | 65.62% | +0.2739 | -0.8747 | 367.2 |
| Seasonal persistence | 0.2787 | 0.5052 | 73.98% | -0.3699 | -0.0809 | 22.8 |

Energy-TTM improved mean MAE by 11.28% and RMSE by 23.50% relative to the baseline, but performed substantially worse on peak timing and peak magnitude. WAPE is high because many synthetic household half-hours have very low consumption. These limitations are exposed rather than recast as an “accuracy” percentage.

## Runtime API and UI

`backend.services.forecast_service.forecast_next_day(client_id, forecast_date)` returns household ID, date and issue time, 48 timestamps, kW and derived kWh values, model identity/version, the exact historical window, actual-comparison status, metrics when legitimate, and `uncertainty=None`.

Consumer **My Energy** offers held-out historical dates and January 1, 2026. Historical actuals are shown only as held-out validation for November–December. January 1 has forecast values only. The Streamlit process loads a cached artifact and never retrains.

## Limitations

This is a synthetic 2025 case study. The model has only aggregate household power as an input, has not been calibrated for uncertainty, and is not a live forecast. One household forecast cannot determine a community DR event; community demand, supply, uncertainty, flexibility, and operational safeguards are future work.
