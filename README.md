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
- Consumer: My Energy, My Household, My Appliances, My DR Participation, Preferences.
- Saved Energy-TTM next-day household forecasts in My Energy: 336 half-hour inputs and 48 half-hour power predictions, with held-out actual comparisons.
- Complete one-minute canonical data and 30-minute operational resampling.
- Parquet predicate/column pushdown and cached portfolio scanning.
- Historical event/participation joins and appliance channels for the 10 panel homes.
- Saved causal multi-task TCN NILM inference in My Appliances for all 50 homes, with quality-aware one-minute estimates, hourly energy, optional panel references, and CSV export.
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

## NILM application integration

The application loads the TCN artifact at `nilm_research/outputs/executed_cpu/tcn/best.pt` by default. Override it
with `NILM_TCN_CHECKPOINT` using a repository-relative path (or an absolute path inside this repository). Select
`NILM_DEVICE=cpu`, `cuda`, or `auto`; `auto` is the default. `NILM_MAX_REQUEST_MINUTES` defaults to 10080 (seven
days). The UI never accepts checkpoint paths and never trains.

Open Consumer Dashboard → My Appliances, select the demo household and `[start, end)` local interval, then choose
Analyze consumption. Source timestamps are interpreted as naive `Africa/Tunis` civil time. Predictions and energy
are shown only for valid contexts; unavailable minutes are null and partial periods are not extrapolated.

See [NILM_INTEGRATION.md](NILM_INTEGRATION.md) for the result contract, quality rules, deployment, limitations, and
checkpoint replacement procedure.

## Forecast training and evaluation

Energy-TTM/Granite-TSFM supports Python 3.10–3.13; use Python 3.13 for the complete application:

```powershell
py -3.13 scripts/train_forecasting.py --households 50 --epochs 5 --batch-size 64
py -3.13 scripts/evaluate_forecasting.py --households 50
```

Artifacts are written to `models/household_forecasting/energy_ttm_v1/`. The UI gives a setup instruction if that compatible artifact is absent.

## Test

```powershell
python -m pytest -q
```

The suite validates every household, canonical completeness, timestamps, units, aggregation, grid balance, event relationships, dashboards, future-model interfaces, and chart contrast.
