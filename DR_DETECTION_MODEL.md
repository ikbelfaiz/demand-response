# AI-assisted DR detection model

## Formulation and issuance boundary

For target day D, the replay is issued at 14:00 on D-1 and produces 48 half-hour forecasts covering D 00:00–24:00. Demand, STEG, and PV observations at or after issuance are unavailable to the models. Actual target-day values, `is_dr_peak`, `is_dr_event`, recorded events, participation, and savings truth are joined only after inference for evaluation.

The existing household Energy-TTM model was reviewed but is not directly reused for community replay. It predicts the 48 points immediately following a contiguous context. At the required 14:00 issuance, those points begin at 14:00 on D-1 rather than midnight on D; reshaping or discarding the intervening ten hours would be invalid. The community pipeline therefore uses deterministic, CPU-friendly multi-output Ridge models appropriate to roughly one year of daily profiles.

## Causal data preparation

The completed canonical grid used by monitoring contains timestamp interpolation based on values on both sides of gaps. That is unsuitable as an unqualified causal backtest input. Training and replay features are built from `data/original_v3_backup/grid_1min.csv`:

- power is averaged into 30-minute slots;
- a slot requires at least 24 of 30 source minutes;
- no two-sided or future-dependent interpolation is performed;
- coverage ratios are retained;
- temperature is omitted because no archived target-day weather forecast exists;
- target-day holiday and Ramadan flags are treated as known calendar information.

Each model uses the 48 observations ending at D-1 13:30, complete D-2 and D-7 profiles, the mean profile from D-8 through D-2, target day-of-week, cyclical month, holiday, and Ramadan. `StandardScaler` is fitted only on the training matrix.

## Models and splits

Three independently scaled multi-output Ridge models predict:

1. `households_consumption_kw`;
2. `steg_production_kw`;
3. `pv_production_kw`.

Training is January 9–May 31, validation is June, and final testing is July–August. This differs from a November–December test because all 36 recorded interventions occur by August; July–August provides 11 held-out operator events and 64 post-DR peak intervals. Ridge regularization is selected using June only. The saved alpha values are 100 for demand, 1 for STEG, and 100 for PV.

## Replay availability and status

The operational selector validates the saved feature contract against the original-source grid. January 9, 2025 is the earliest possible target because the mean-profile feature reaches back to D-8; December 31 is the latest target in the dataset. There are 327 supported days. Required lag profiles cross source intervals below 80% coverage on February 5–12, March 4–10, September 17–23, and October 18–25, so those dates remain unavailable rather than being imputed for inference.

Only July 1–August 31 is explicitly proven by artifact metadata to be excluded from training and model selection, and is labeled **Historical replay — out-of-sample evaluation**. January–June and September–December predictions are labeled **Retrospective prediction — this date may have been used during model development**. The latter status is conservative: availability does not imply evaluation validity.

Historical DR changes the observed process. The generator first constructs natural demand, schedules events using a noisy day-ahead margin forecast, then reruns households with compliant AC setpoint changes and deferred water-heater load. Feeder demand and `is_dr_peak` are post-intervention. Demand target days intersecting recorded interventions are excluded from training and validation. Prior observed intervention effects can remain in causal context, which is an unavoidable operational-history limitation.

## Supply forecasts

The dataset has realized allocated STEG and PV, not a future dispatch plan or archived solar forecast. Separate past-only profile regressors therefore forecast both components. Predicted total supply and margin are:

```text
predicted_supply_kw = predicted_steg_kw + predicted_pv_kw
predicted_margin_kw = predicted_supply_kw - predicted_demand_kw
```

These are synthetic neighborhood allocations, not national STEG emergency forecasts.

## Risk and decision policy

A risk slot has a negative predicted margin. Consecutive slots are grouped and described by start/end, maximum shortfall, duration, and shortage energy. No probability is reported because no probability calibration was performed.

The separate `research-policy-v1` recommends an event only when a risk has at least 1 kW maximum deficit, 60 minutes duration, and 1 kWh shortage energy. Events are capped at 120 minutes, separated by 60 minutes, and limited to two per day. Recommended reduction equals the maximum forecast shortfall. June validation produced no predicted negative-margin windows, so these preregistered engineering thresholds were retained rather than tuned on final-test labels. They are experimental settings, not STEG policy.

## Artifacts and commands

Artifacts are stored in `models/community_dr/profile_ridge_v1/`, including models, scalers, metadata, model selection, decision policy, slot-level replay results, recommendations, and summaries.

```powershell
py -3.13 scripts/train_community_forecast.py
py -3.13 scripts/evaluate_dr_detection.py
```

The application loads these saved artifacts and does not retrain during navigation.
