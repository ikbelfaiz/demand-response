# Consumer D+1 Household Forecast Improvement

Date: 2026-10-08  
Scope: Consumer Dashboard household forecasts only  
Operator/community artifact before work: `3138cbb6d73ed848c3b29f48b2801a0a9ebc434db0baab8f311253907150863e`

## Outcome

The existing consumer Energy-TTM artifact remains active. Two newly trained Energy-TTM candidates and three fixed calibration blends were evaluated, but none met the activation rule declared before final evaluation: at least 2% lower validation RMSE and MAE, with no worse WAPE or peak-interval RMSE. Candidate weights were saved separately and the original artifact was not overwritten.

The Consumer Dashboard was improved independently of model activation. It shows model version, actual and forecast daily energy, MAE, RMSE, WAPE, absolute peak-timing error, and an explicit warning that appliance events may not be predictable from aggregate demand alone. Persistence remains an offline evaluation baseline and is intentionally not exposed in the operational interface.

The Operator forecast, DR detector, decision engine, inference path, and artifact were not modified or retrained.

## Original consumer model

- Architecture: `TinyTimeMixerForPrediction` initialized from `EnergyFM/energy-ttm`.
- Shape: one input channel, 336 half-hours (seven days), direct 48-half-hour output.
- Configuration: seven non-overlapping 48-slot patches, `d_model=16`, three encoder mixer layers, eight decoder layers, MSE objective.
- Training: all 50 households, 13,300 daily training windows, five epochs, AdamW at 0.001.
- Normalization: mean/std calculated independently from each forecast's 336 historical values and used for inverse scaling. This is already household/origin-specific and is consistent between training and inference.
- Household identification: no client ID, static household embedding, or household-specific parameters enter Energy-TTM. The model sees standardized aggregate history only. It therefore learns a shared mapping across all households and cannot explicitly distinguish two households with similar normalized contexts.
- Calendar/weather: not supported by the current univariate wrapper. No weekday, holiday, Ramadan, occupancy, appliance-state, or archived forecast-weather feature is passed to Energy-TTM.

The prior audit verified that all 278 saved tensors changed during v3 fine-tuning, so poor forecasts are not caused by an untrained artifact.

## Leakage-safe experimental data

Candidates and comparisons read the original backup household parquets rather than the whole-year two-sided completed copies:

- `data/original_v3_backup/households_1min_C001_C025.parquet`
- `data/original_v3_backup/households_1min_C026_C050.parquet`

Source power is converted from W to kW and averaged at 30-minute resolution. A target interval is accepted with at least 24/30 observed source minutes. Lower-coverage context gaps are filled only from the most recent past operational observation; future observations are never used. A day containing any target interval below 80% coverage is excluded rather than fabricated.

This retained:

| Split | Possible windows | Eligible windows | Excluded incomplete targets |
|---|---:|---:|---:|
| Train, 2025-01-08–2025-09-30 | 13,300 | 13,097 | 203 |
| Validation, 2025-10-01–2025-10-31 | 1,550 | 1,529 | 21 |
| Test, 2025-11-01–2025-12-31 | 3,050 | 3,001 | 49 |

The same leakage-safe inputs and observed targets were used for the current model, candidates, and every baseline during comparison.

## Baselines

All baselines are household-specific and causal:

- Previous-day persistence: D-1 profile.
- Previous-week persistence: D-7 profile.
- Historical slot median: median of each half-hour over all prior days.
- Weekday/weekend profile: half-hour median over prior days of the same weekday/weekend class.

No test observation is used to construct a prediction.

## Experiments

### Unsupported patch experiment

Restoring the checkpoint's native 24-slot patch while keeping the required 336-slot context creates 14 patches. The downloaded checkpoint's patch-mixer is structurally fixed to seven patches; its first forward pass failed with a 14×7 matrix mismatch before any optimization step. This experiment was rejected rather than silently shortening the seven-day context or changing architecture family.

### Trained candidates

Both supported candidates keep 336→48 direct Energy-TTM forecasting, use 48-slot patches, train all 50 households, use AdamW at 0.0003, and run up to ten epochs with validation-based checkpoint selection.

| Candidate | Training change | Best normalized validation MSE | Selected epoch |
|---|---|---:|---:|
| `energy_ttm_causal_mse` | Leakage-safe targets, lower learning rate, 10 epochs | 0.654926 | 10 |
| `energy_ttm_causal_peak` | Same, plus 3× MSE weight on six highest target slots | 0.728922 | 10 |

Validation loss was still declining at epoch ten for both candidates, so they do not show classical overfitting. More epochs might help, but the experiment provides no evidence that epochs alone would solve household peak timing; that claim is not made.

### Validation results

Metrics are means over 1,529 household-days. Energy error and peak errors shown here are absolute; signed bias remains available in the CSV outputs.

| Method | MAE kW | RMSE kW | WAPE | Daily energy abs. error kWh | Peak magnitude abs. error kW | Peak timing abs. error min | Top-six peak-slot RMSE kW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Energy-TTM | 0.18658 | 0.29244 | 57.91% | 1.2147 | 0.7239 | 277.7 | 0.6344 |
| Causal MSE candidate | 0.18325 | 0.29173 | 56.74% | 1.2049 | 0.7667 | 279.2 | 0.6388 |
| Peak-aware candidate | 0.21996 | 0.31523 | 69.49% | 2.2374 | 0.6307 | 286.6 | 0.5362 |
| Energy-TTM + 25% weekday/weekend profile | 0.17736 | 0.28736 | 55.02% | 1.1495 | 0.6836 | 275.5 | 0.6260 |

The causal MSE candidate improved overall errors only marginally and worsened peak errors. Peak weighting improved top-six peak-slot RMSE by 15.5% but substantially worsened ordinary demand and energy metrics. The profile calibration was best overall, but its RMSE improvement was 1.74%, below the declared 2% activation threshold. It was not deployed.

## Untouched test-period results

Because no learned candidate passed validation, only the current Energy-TTM and causal baselines were reported on the 3,001 eligible test household-days.

| Method | MAE kW | RMSE kW | WAPE | Daily energy abs. error kWh | Peak magnitude abs. error kW | Peak timing abs. error min | Top-six peak-slot RMSE kW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Energy-TTM | 0.24599 | 0.38546 | 65.32% | 2.0459 | 0.9130 | 409.6 | 0.8115 |
| Previous day | 0.29096 | 0.52170 | 77.23% | 2.3368 | 0.6523 | 391.0 | 0.9716 |
| Previous week | 0.27748 | 0.50427 | 73.71% | 2.5262 | 0.7014 | 383.3 | 0.9724 |
| Historical slot median | 0.22518 | 0.42311 | 57.33% | 2.6877 | 0.9385 | 416.1 | 1.0151 |
| Weekday/weekend profile | 0.20505 | 0.38956 | 52.22% | 2.4231 | 0.9342 | 411.6 | 0.9046 |

Energy-TTM retains the best test RMSE and peak-slot RMSE among activated/baseline methods, while the weekday/weekend profile has better MAE and WAPE. December is harder than November: Energy-TTM RMSE rises from 0.3657 to 0.4045 kW and absolute daily-energy error from 1.7272 to 2.3535 kWh.

## Household error analysis

Worst test RMSE households:

| Household | RMSE kW | WAPE | Peak-slot RMSE kW | Peak timing error min |
|---|---:|---:|---:|---:|
| C033 | 0.5579 | 77.30% | 1.1494 | 374.2 |
| C021 | 0.5494 | 82.65% | 1.0925 | 308.6 |
| C047 | 0.5374 | 75.79% | 1.0729 | 364.7 |
| C037 | 0.5265 | 70.54% | 1.0674 | 366.9 |
| C035 | 0.5182 | 75.05% | 1.0980 | 412.1 |

Across households, RMSE correlates strongly with held-out load variability (`r=0.981` with standard deviation) and high-load magnitude (`r=0.965` with P95). This supports the diagnosis that the shared model collapses toward a smooth daily profile and misses irregular high-power events. Absolute error also naturally scales with household load (`r=0.953` with mean kW).

## Diagnosis

- **Incorrect scaling:** not found. Scaling is per household forecast origin and inverse scaling is correct.
- **Incorrect household ID:** no wrong ID is passed, but the architecture receives no ID at all. All households share one mapping.
- **Training/inference mismatch:** not found in the active pipeline. Candidate training/evaluation also share the same causal preparation.
- **Underfitting:** supported. Both training curves continued to improve, current and causal models systematically under-predict peaks, and forecasts are smoother than high-variance targets.
- **Overfitting:** not supported by validation curves.
- **Insufficient training:** plausible but not sufficient as an explanation. Ten lower-rate epochs produced only marginal ordinary accuracy improvement.
- **Peak objective:** helps peak intervals but creates material energy/profile tradeoffs. It should not replace the general model without a multi-objective validation rule.
- **Calendar effects:** the causal weekday/weekend baseline is competitive, demonstrating useful calendar signal that the univariate wrapper cannot directly consume.
- **Intrinsic uncertainty:** exact appliance starts are weakly determined by aggregate history alone. The 409.6-minute mean peak-timing error, shared by learned and profile methods, is strong evidence that deterministic point forecasting cannot reliably time individual events with current inputs.
- **Two-sided completion:** avoided in candidate training and fair comparisons. It remains a provenance limitation of the original active model's training corpus, which is another reason not to claim that the marginal candidate result establishes a definitive architecture ranking.

## Dashboard changes

The consumer forecast section now provides:

- Actual and Energy-TTM traces.
- Forecast and actual daily energy where actual data exists.
- MAE, RMSE, WAPE, daily energy error, and absolute peak timing error.
- Active model version.
- A clear point-forecast/appliance-event limitation notice.

No confidence band is fabricated.

## Artifacts and reproduction

- Candidate trainer: `scripts/train_consumer_forecast_candidates.py`
- Selection/evaluation: `scripts/evaluate_consumer_forecast_candidates.py`
- Leakage-safe preparation: `backend/intelligence/consumer_forecast_data.py`
- Candidate artifacts: `models/household_forecasting/candidates/`
- Detailed results: `outputs/consumer_forecast_improvement/`
- Activation decision: `outputs/consumer_forecast_improvement/selection_stage2.json`

The scientifically justified result is retention, not forced deployment: the experiments clarified where accuracy is lost, improved consumer diagnostics, established stronger baselines, and produced isolated candidate artifacts for future work without degrading the active or operator systems.
