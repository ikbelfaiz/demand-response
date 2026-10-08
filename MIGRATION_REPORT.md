# Dataset-Centered Platform Redesign Report

## Outcome

The application is now an **Intelligent Energy Monitoring and Demand Response Decision Support Platform**, not a household scheduling simulator. Its runtime uses only `data/dr_dataset_2025_1min_v2.csv`. The legacy `household_energy.csv` remains an unreferenced archival file and is not loaded, mapped or mentioned by the operational interface.

The prior three-tab layout was replaced with five sidebar workspaces: Energy Overview, Household Analysis, Regional Grid, DR Intelligence and Data Quality.

## Verified dataset

| Property | Result |
|---|---|
| Shape | 525,600 rows × 15 source columns |
| Coverage | 2025-01-01 00:00 to 2025-12-31 23:59 |
| Frequency | Continuous one-minute timestamps |
| Timestamp timezone | Not supplied; retained as naive civil time |
| Duplicate timestamps | 0 |
| Missing timestamp intervals | 0 |
| Household identity | `C001` |
| Region identity | `TUN` |
| Household measurement | aggregate power and AC/water-heater/washing-machine power in W |
| Regional measurements | zone demand, STEG production and PV production in MW |
| Historical labels | `is_dr_event`, `is_dr_peak` |
| Price/tariff | absent |

The dataset contains 35 supplied `is_dr_event` windows of 120 minutes each. They start at several different hours between 18:00 and 22:00. These are retained as historical labels only and are never interpreted as current predictions.

## Missing-data audit

| Source channel | Missing rows | Longest outage (minutes) |
|---|---:|---:|
| `aggregate_power_w` | 1,739 | 563 |
| `ac_power_w` | 2,164 | 1,328 |
| `water_heater_power_w` | 1,037 | 302 |
| `washing_machine_power_w` | 4,398 | 3,859 |
| `temperature_c` | 2,280 | 60 |
| `zone_consumption_mw` | 277 | 75 |
| `pv_production_mw` | 151 | 30 |
| `is_dr_peak` | 427 | 75 |

The analysis copy time-interpolates only internal continuous-sensor gaps up to 180 minutes. Long outages stay missing. Provenance flags preserve original missingness. Labels are not interpolated or zero-filled. Energy summaries expose valid-observation coverage, and incomplete totals are labeled as observed energy rather than unqualified complete consumption.

## Household versus regional scope

- `aggregate_power_w / 1000` is household demand in kW.
- Appliance channels are household kW after conversion, but are not assumed to constitute all household demand.
- `zone_consumption_mw`, `steg_production_mw` and `pv_production_mw` remain regional MW signals.
- Regional production minus regional demand is shown only as an explicitly partial reported-signal comparison. No grid deficit, emergency or adequacy conclusion is inferred because imports, other generation, storage, losses and balancing actions may be absent.
- Household peaks are descriptive and never used as automatic DR activations.

## Removed legacy functionality

- Fixed daily 18:00–20:00 DR window
- Household threshold-based DR declarations
- Household flexible-fraction scenario and before/after curve
- Appliance shifting and hardcoded AC curtailment
- Simulated savings, shifted-energy and optimization cards
- Synthetic tariffs, electricity cost and cost-savings calculations
- Household rooftop solar, self-consumption and grid import/export calculations
- Dishwasher and synthetic base-load assumptions
- Scenario recommendation rules
- Three-tab dashboard and its legacy chart guides/controls

No disabled placeholders for these features remain in the redesigned interface.

## Current page responsibilities

| Page | Implemented behavior |
|---|---|
| Energy Overview | Household KPIs and trends plus separately identified regional demand/production/PV views |
| Household Analysis | Load curve, hourly profile, heatmap, distribution, daily energy, top readings and three measured appliances |
| Regional Grid | Regional demand/production/PV, daily trends, top demand readings and qualified signal comparison |
| DR Intelligence | Empty model status, future pipeline, capability status and optional historical-label exploration |
| Data Quality | Schema, identifiers, coverage, missing statistics, outages, descriptive statistics, preview/export |

## AI-ready architecture

`backend/intelligence` defines interfaces and provenance-carrying result objects for:

- standardized monitoring features;
- peak-demand forecasting;
- candidate grid-condition detection;
- separately governed DR activation decisions;
- household flexibility estimation;
- DR optimization plans;
- versioned model registration/loading;
- held-out detection evaluation.

The registry is empty by default. Monitoring pages do not instantiate a detector and do not read historical event labels. Feature construction explicitly excludes `dr_event`. Therefore a source label cannot be accidentally displayed as an AI prediction.

## Remaining limitations

- No trained model, validated grid-stress definition or activation policy exists.
- The CSV lacks timezone metadata.
- No verified tariff supports monetary analysis.
- Regional signal coverage may not represent a complete supply-demand balance.
- No household PV or bidirectional meter supports household import/export analysis.
- Appliance flexibility constraints and ground truth are unavailable.
- Causal DR impact requires a validated counterfactual methodology.
- Historical label provenance and activation criteria require external documentation before supervised modeling.
