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
- Saved causal multi-task TCN NILM inference in My Appliances for all 50 homes, with quality-aware one-minute estimates, hourly energy, optional panel references, and CSV export.
- Customer profiling (Operator ▸ Customer Profiling, Consumer ▸ My Household / My DR Participation): K-means segments trained January–mid-June, acceptance rates, typical curves by season and day type, priority list and chatbot context for each peak window, event costs (CPP +50 %), segment savings and rewards. Train with `python scripts/train_customer_profiling.py`; see [CUSTOMER_PROFILING.md](CUSTOMER_PROFILING.md).
- Explicit offline forecast training/evaluation; Streamlit only loads versioned artifacts and never retrains.

## Historical forecast availability

Consumer **My Energy** provides a historical-only date selector accepting every target day from **2025-01-08 through 2025-12-31** for each of the 50 households. January 8 is the first date with the required 336 half-hour observations from January 1–7. The selector defaults to December 31 and does not expose a post-dataset January 2026 mode. The application derives these bounds from the installed household data and keeps the 48-slot Energy-TTM contract unchanged.

Operator **DR Detection & Forecasting** derives replay availability from the saved model's actual D-1 14:00 features. The overall range is **2025-01-09 through 2025-12-31**. Original-source gaps make these internal ranges unavailable: February 5–12, March 4–10, September 17–23, and October 18–25. Unsupported dates display an explanation rather than fabricating lag profiles. The page keeps model-registry details out of the operational interface while continuing to load the same saved artifact.

Dates explicitly declared by artifact metadata as test data are labeled **Historical replay — out-of-sample evaluation**. Training, validation, or otherwise unproven dates are labeled **Retrospective prediction — this date may have been used during model development**. Consumer held-out dates are November–December; operator held-out dates are July–August. Forecast availability is therefore broader than evaluation validity.

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

## Community DR historical replay

The community pipeline is separate from the household forecaster. It uses causal multi-output Ridge profile models because the 14:00 issuance cutoff creates a ten-hour gap before the next calendar day, which is incompatible with directly applying the contiguous Energy-TTM household horizon. Training uses the original pre-completion grid with past-only aggregation and excludes historical intervention target days from demand training.

```powershell
py -3.13 scripts/train_community_forecast.py
py -3.13 scripts/evaluate_dr_detection.py
py -3.13 -m streamlit run app.py
```

Saved models and backtest outputs are under `models/community_dr/profile_ridge_v1/`. See [DR_DETECTION_MODEL.md](DR_DETECTION_MODEL.md) and [DR_BACKTEST_REPORT.md](DR_BACKTEST_REPORT.md). Recommendations are historical simulations only and never write to `dr_events.csv`.

## DR orchestrator and chatbot

The orchestrator (`backend/orchestration/`) turns the community D+1 forecast into critical peak pricing events: proposal at 14:30 the day before, operator confirmation before the 18:00 alerts, personalised messages (FR / AR / EN) using customer profiling and NILM, a reminder during the event, and the bill and summary the next morning.

```powershell
python scripts/run_orchestrator_replay.py --start 2025-06-01 --end 2025-09-30   # whole season, simulated answers
python -m streamlit run app.py
```

In the app, the data is historical, so the orchestrator runs on a **demo clock**:

1. *Operator Dashboard ▸ DR Orchestrator* → **Build history** (replays June → the chosen date with simulated household answers).
2. **Next forecast peak** moves the clock to the next 14:30 decision with a forecast peak → **Confirm** (or change the level / **Veto**).
3. **Next step** → 18:00: alerts are sent.
4. *Consumer Dashboard ▸ My DR Assistant* (same household as in the priority list): notification card with **Accept / Decline / Later**, and a chat that answers questions about the window, the price, what to switch off and the expected extra cost.
5. Back on the operator page, **Next step** runs the event (reminder after the first half-hour) and the bills the next morning; *My DR Participation* shows the extra cost of each event.

State is stored in `outputs/orchestrator/app.db` (git-ignored). The smart-meter readings are historical: answering *Accept* in the demo does not change the consumption that is billed.

## Test

```powershell
python -m pytest -q
```

The suite validates every household, canonical completeness, timestamps, units, aggregation, grid balance, event relationships, dashboards, future-model interfaces, and chart contrast.
