# Demand Response v3

A Streamlit research platform for a **synthetic 50-household Tunis neighborhood case study**. It provides operator and selected-household perspectives. It is not an operational STEG system or evidence of real grid shortages.

## Run

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m streamlit run app.py
```

The application uses one canonical dataset: `data/dr_dataset_v3_grid_et_code/grid_1min.csv` and the two household parquet files directly under `data/`. These synthetic sources were deterministically completed for continuous demonstration charts. Their pre-completion versions and SHA-256 records are retained in `data/original_v3_backup/`. Reproduce the migration with:

```powershell
python scripts/complete_v3.py
```

## Implemented

- Operator: Community Overview, Community Grid, DR Events, Household Analytics.
- Operator AI research: DR Detection & Forecasting with D-1 14:00 historical replay, community demand/supply forecasts, risk windows, and simulated recommendations.
- Consumer: My Energy, My Household, My Appliances, My DR Participation.
- Saved Energy-TTM next-day household forecasts in My Energy: 336 half-hour inputs and 48 half-hour power predictions, with held-out actual comparisons.
- Complete one-minute canonical data and 30-minute operational resampling.
- Parquet predicate/column pushdown and cached portfolio scanning.
- Historical event/participation joins and appliance channels for the 10 panel homes.
- Explicit offline forecast training/evaluation; Streamlit only loads versioned artifacts and never retrains.

## Scientific conventions

- Power is averaged during resampling. Half-hour energy is `mean kW × 0.5 h`.
- Canonical installed sensors are complete at one-minute resolution.
- The 40 non-panel homes retain null appliance columns because those channels are structurally unavailable; no appliance readings were fabricated.
- Feeder demand already includes the synthetic 3% network-loss assumption.
- Available supply is a calibrated neighborhood allocation after the generator’s assumed 35% non-residential share.
- Historical DR labels are not application predictions.
- Timestamps have no timezone metadata and remain naive local civil time.
- Synthetic counterfactual truth remains isolated in the research evaluation module.

See [FORECASTING_MODEL.md](FORECASTING_MODEL.md), [V3_DATA_DICTIONARY.md](V3_DATA_DICTIONARY.md), [ARCHITECTURE.md](ARCHITECTURE.md), [MIGRATION_REPORT.md](MIGRATION_REPORT.md), and [DATA_COMPLETION_REPORT.md](DATA_COMPLETION_REPORT.md).

## Forecast training and evaluation

Energy-TTM/Granite-TSFM supports Python 3.10–3.13; use Python 3.13 for the complete application:

```powershell
py -3.13 scripts/train_forecasting.py --households 50 --epochs 5 --batch-size 64
py -3.13 scripts/evaluate_forecasting.py --households 50
```

Artifacts are written to `models/household_forecasting/energy_ttm_v1/`. The UI gives a setup instruction if that compatible artifact is absent.

## Community DR historical replay

The community pipeline is separate from the household forecaster. It uses causal multi-output Ridge profile models because the 14:00 issuance cutoff creates a ten-hour gap before the next calendar day, which is incompatible with directly applying the contiguous Energy-TTM household horizon. Training uses the original pre-completion grid with past-only aggregation and excludes historical intervention target days from demand training.

```powershell
py -3.13 scripts/train_community_forecast.py
py -3.13 scripts/evaluate_dr_detection.py
py -3.13 -m streamlit run app.py
```

Saved models and backtest outputs are under `models/community_dr/profile_ridge_v1/`. See [DR_DETECTION_MODEL.md](DR_DETECTION_MODEL.md) and [DR_BACKTEST_REPORT.md](DR_BACKTEST_REPORT.md). Recommendations are historical simulations only and never write to `dr_events.csv`.

## Test

```powershell
python -m pytest -q
```

The suite validates every household, canonical completeness, timestamps, units, aggregation, grid balance, event relationships, dashboards, future-model interfaces, and chart contrast.
