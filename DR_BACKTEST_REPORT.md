# DR historical replay backtest

## Evaluation design

The final test covers 62 days from July 1 through August 31, 2025. Every forecast is issued at 14:00 on the prior day. July–August was selected before final evaluation because it is the last chronological period containing a useful number of recorded interventions: 11 operator events. June is the validation period and is not included in reported test metrics.

Seasonal persistence predicts each target profile from the same half-hour slots seven days earlier. Its negative-margin rule is the deterministic DR baseline. Recorded `is_dr_peak` is evaluated as a post-intervention grid-risk label; `is_dr_event` and `dr_events.csv` are evaluated separately as historical activation behavior.

## Forecast results

Values below are means of daily metrics across 62 executed replays.

| Target | Method | MAE kW | RMSE kW | WAPE | Daily energy error kWh | Peak magnitude error kW | Absolute peak timing error min |
|---|---|---:|---:|---:|---:|---:|---:|
| Demand | Ridge | 12.628 | 16.159 | 38.50% | -170.85 | -0.471 | 169.84 |
| Demand | Persistence | 10.976 | 13.901 | 36.97% | +29.37 | +1.884 | 106.94 |
| STEG | Ridge | 4.259 | 4.876 | 4.40% | -28.50 | -1.597 | 89.03 |
| STEG | Persistence | 6.301 | 6.885 | 6.54% | -10.50 | -0.147 | 133.55 |
| PV | Ridge | 1.608 | 2.482 | 20.67% | -21.08 | -2.343 | 30.97 |
| PV | Persistence | 1.289 | 1.969 | 20.67% | +5.32 | +0.341 | 26.61 |
| Total supply | Ridge | 4.916 | 5.759 | 4.58% | -49.59 | -3.681 | 87.10 |
| Total supply | Persistence | 6.946 | 7.822 | 6.51% | -5.17 | -0.267 | 73.06 |

The trained model improved STEG and total-supply MAE/RMSE but did not beat persistence for demand or PV. The demand model underpredicted daily energy materially during the held-out summer period.

## Detection and recommendation results

| Comparison | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| Forecast negative margin vs post-DR peak | 0.471 | 0.129 | 0.203 | 9 | 54 |
| Recommended interval vs post-DR peak | 0.438 | 0.113 | 0.179 | 9 | 55 |
| Persistence margin rule vs post-DR peak | 0.273 | 0.242 | 0.256 | 40 | 47 |
| Recommendation vs historical activation | 0.375 | 0.136 | 0.200 | 10 | 38 |

The policy produced seven recommended events. After excluding 36 source-affected label slots, the remaining labels form 13 post-DR peak windows; event precision was 0.429 and recall 0.231, with 0.065 false alerts per replay day and 16.15 minutes mean best overlap. Against 11 historical operator events, event recall was 0.273. Mean forecast-to-event lead time was 30.36 hours.

The proposed pipeline did not outperform the baseline detector F1. It traded fewer false alerts for substantially lower recall. This is a research result, not evidence of deployment readiness.

## Counterfactual and label limitations

Ground-truth savings is limited to historical event totals per household. In the held-out period it is used only to summarize natural-versus-observed event energy; it cannot reconstruct a reliable 30-minute natural-demand curve for arbitrary recommended windows. It is never a model input.

The 550 held-out event-household records contain 1,786.388 kWh natural baseline energy and 1,585.200 kWh observed energy, a synthetic 201.188 kWh reduction. These totals describe recorded interventions only and are not used to score arbitrary model recommendations.

`is_dr_peak` was generated from the post-response feeder curve, while operator events were scheduled from noisy forecasts of natural demand. Consequently, successful intervention can suppress the label that a detector is being evaluated against. Historical events also contain false alarms and misses. Results therefore report peak-risk detection and historical-activation matching separately.
