# Intelligent Energy & Demand Response Platform

A Streamlit energy-monitoring and research platform built exclusively around `data/dr_dataset_2025_1min_v2.csv`. It separates household measurements, regional grid indicators, historical dataset labels, and future AI outputs so that none are presented as something they are not.

The application currently performs monitoring, aggregation, peak analysis and data-quality reporting. It does **not** contain a trained forecasting model, DR detector, activation policy, flexibility estimator or optimizer.

## Application pages

1. **Energy Overview** — household energy/demand/load factor alongside separately identified regional demand, STEG production and regional PV. Includes daily and hourly patterns.
2. **Household Analysis** — C001 load curve, peak readings, hourly profile, heatmap, demand distribution and measured AC, water-heater and washing-machine channels.
3. **Regional Grid** — TUN demand, STEG production, PV production, regional trends and peak-demand observations. Production-demand arithmetic is explicitly a partial reported-signal comparison, not an adequacy or emergency assessment.
4. **DR Intelligence** — reports that no detection model is implemented, explains the future detection/decision pipeline, and optionally exposes CSV event labels as historical data only.
5. **Data Quality** — schema, coverage, missing values, sensor outages, descriptive statistics, filtered preview and CSV export.

All pages share a sidebar period selector supporting day, week, month and custom ranges.

## Project structure

```text
├── app.py
├── backend/
│   ├── config.py                    # sole dataset registration and semantic mapping
│   ├── data/                        # loading, validation, schema, bounded preprocessing
│   ├── services/
│   │   ├── analytics_service.py
│   │   ├── energy_service.py
│   │   ├── household_service.py
│   │   ├── regional_service.py
│   │   ├── data_quality_service.py
│   │   └── historical_event_service.py
│   ├── intelligence/
│   │   ├── feature_engineering.py
│   │   ├── forecasting.py
│   │   ├── event_detection.py
│   │   ├── decision_engine.py
│   │   ├── evaluation.py
│   │   └── model_registry.py
│   └── utils/time_utils.py
├── frontend/
│   ├── pages/                       # five page renderers
│   ├── charts.py
│   ├── components.py
│   └── styles.py
├── data/dr_dataset_2025_1min_v2.csv
├── tests/
└── MIGRATION_REPORT.md
```

`app.py` owns configuration, navigation and service-to-page routing. Scientific calculations remain in backend services; frontend modules render their outputs.

## Dataset and units

The CSV has 525,600 unique one-minute timestamps covering `2025-01-01 00:00` through `2025-12-31 23:59`. Timestamps contain no timezone information. Every row belongs to household/client `C001` and region `TUN`.

| Source field | Standardized use | Unit |
|---|---|---:|
| `aggregate_power_w` | `household_power_kw` | W converted to kW |
| `ac_power_w` | `ac_power_kw` | W converted to kW |
| `water_heater_power_w` | `water_heater_power_kw` | W converted to kW |
| `washing_machine_power_w` | `washing_machine_power_kw` | W converted to kW |
| `zone_consumption_mw` | `zone_demand_mw` | MW |
| `steg_production_mw` | `system_production_mw` | MW |
| `pv_production_mw` | `zone_pv_production_mw` | MW |
| `temperature_c` | `temperature_c` | °C |
| `is_dr_event` | historical event label | binary |
| `is_dr_peak` | historical peak label | binary/unknown |

The regional MW signals are never mixed numerically with household kW measurements. Regional PV is not treated as household rooftop solar.

## Data-processing rules

- The source CSV remains unchanged.
- Missing values are never silently filled with zero.
- Original missingness is retained in `*_was_missing` fields.
- Internal continuous-sensor gaps of at most 180 minutes may be time-interpolated in the analysis copy. Longer outages remain missing and therefore appear as chart gaps.
- Missing `is_dr_peak` values remain unknown; labels are not interpolated.
- Energy is integrated at native resolution using timestamp-derived forward durations. Bounded imputed readings are counted and disclosed; unresolved missing readings contribute no invented energy, and original-observation coverage is reported with period totals.
- Display resampling uses mean power, never summed kW. Native one-minute data remains the basis for metrics.
- Historical `is_dr_event` values are accessible only through the historical-label service and optional DR Intelligence exploration. They do not control the operational dashboard.

See [MIGRATION_REPORT.md](MIGRATION_REPORT.md) for the audit and removed legacy behavior.

## Installation and launch

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The configured CSV path is resolved from the project root, independently of the terminal working directory.

## Future AI integration

The `backend/intelligence` contracts keep these stages distinct:

```text
validated data → versioned features → forecast/detection result
               → separate activation decision → flexibility estimate → optimization plan
               → held-out label evaluation
```

A future integration should:

1. Implement the relevant abstract interface (`Forecaster`, `EventDetector`, `DecisionEngine`, `FlexibilityEstimator` or `DROptimizer`).
2. Register an already-trained artifact using an explicit name and version.
3. Return timestamped result objects with model/policy provenance.
4. Keep `is_dr_event` out of operational feature construction unless an explicitly documented supervised-training pipeline uses a training-only label copy.
5. Evaluate frozen predictions against held-out labels through `DetectionEvaluator`.
6. Label model outputs as predictions and activation outputs as recommendations/decisions.

No placeholder algorithm is registered by default, so the platform runs without a trained model and cannot accidentally emit a fabricated event prediction.

## Tests

```powershell
.\.venv-energy-ttm\Scripts\python.exe -m pytest -q
```

Tests cover timestamps, units, missing intervals, bounded interpolation, energy and coverage, time filtering/aggregation, household and regional analytics, appliance discovery, outages, historical-label separation, feature construction and empty model-registry behavior. Streamlit page smoke tests are also performed during development.
