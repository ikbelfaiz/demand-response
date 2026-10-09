# Architecture

## Data flow

```text
canonical complete v3 sources (1 minute)
  └─ resampling service → 30-minute dashboard layer
       ├─ community and grid services → operator pages
       └─ household and participation services → consumer pages
```

`backend/config.py` contains root-relative canonical paths. `backend/data/v3_loader.py` owns cached CSV access and parquet pushdown. `aggregation.py` enforces power/energy semantics. `validator.py` checks schema metadata and relationships.

`scripts/complete_v3.py` verifies or creates backups, completes synthetic sensor channels, validates temporary outputs, and atomically replaces canonical files. Backup checksums and completion counts are stored in `data/original_v3_backup/`. The script is a reproducibility tool, not a runtime dependency.

Services own community, grid, household, event, participation, segmentation, and forecast calculations. Frontend modules compose them with a centralized accessible Plotly theme. `app.py` handles navigation, period selection, household selection, and caching. Data-quality validation remains internal even though the Operator Data Explorer UI was removed.

## Performance

The grid is loaded once. Household queries touch one parquet half and push client, timestamp, and selected columns into PyArrow. Portfolio summaries batch-scan only household ID and aggregate power and are cached by period. Plots operate on 30-minute data, and table/export previews are bounded.

## Forecasting boundary

- `forecast_features.py` constructs exact seven-day contexts and rejects missing slots; target values never enter context normalization.
- `forecasting.py` lazily loads a versioned TinyTimeMixer artifact. The adapted 48-point patcher and head are trained offline; page loads never train.
- `forecast_service.py` derives each household's eligible dates from installed data and returns power, derived interval energy, model metadata, input provenance, KPIs, and error metrics only when artifact provenance identifies a held-out target.
- Consumer availability is January 8–December 31, 2025. November–December are explicitly held out; earlier predictions are retrospective and never presented as independent evaluation.

## Customer profiling boundary

- `customer_profiling.py` receives the forecast peak window and deficit; it produces segments and acceptance rates, typical curves, the priority list, the chatbot context, and the event economics (costs, segment savings, rewards). It never reads synthetic truth.
- `scripts/train_customer_profiling.py` trains offline and writes `models/customer_profiling/kmeans_v1/`; `segmentation_service.py` only loads those artifacts and falls back to quartiles when absent.

## Remaining intelligence boundary

- NILM defines panel targets, household-wise splitting, and a model-loading boundary.
- Baseline inference cannot import research-only ground truth.
- The orchestrator reports forecast availability from the artifact registry; DR decision and activation capabilities remain disabled.
- Historical event labels cannot trigger decisions; the model registry starts empty.

## Community historical replay

`community_forecasting.py` reads the original pre-completion grid and creates 30-minute values directly from available source minutes. It retains coverage metadata and never uses the canonical two-sided completion as a causal model input. Target-day weather is omitted because no archived forecast exists.

At D-1 14:00, separate saved Ridge models forecast the 48 target-day profiles for community demand, allocated STEG supply, and PV. Inputs are the prior 24 hours ending at issuance, D-2 and D-7 profiles, the mean of D-8 through D-2, and known calendar features. Each scaler is fitted on training data only.

`eligible_community_forecast_dates()` validates those dependencies against the original-source operational grid. The first possible replay is January 9 because D-8 must exist. Availability continues through December 31, excluding internal dates whose required source lags fall below 80% coverage. Only the metadata-declared July–August test interval is labeled out of sample; all other dates are conservative retrospective demonstrations.

`peak_detection.py` groups consecutive negative forecast margins. `decision_engine.py` applies a separate versioned research policy. `historical_replay.py` joins forecasts, actual post-intervention observations, peak labels, and historical operator events only after predictions exist. `backtesting.py` evaluates interval and event matching. Ground-truth savings remain accessible only through the research evaluation accessor.

## State and integrity

Canonical synthetic files are the sole runtime source. Pre-completion backups are not loaded by the app. Optional onboarding inputs and mock responses live only in Streamlit session state. The perspective selector is not authentication.
