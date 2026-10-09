# Customer profiling

Customer profiling receives the peak window and the deficit from the D+1 forecast, decides who to target, gives the chatbot each household's context, and computes the economics of each event. It uses only the 30-minute smart-meter reading and the temperature.

```bash
python scripts/train_customer_profiling.py              # train + all outputs (~20 s)
python scripts/train_customer_profiling.py --evaluate   # + performance tests -> outputs/customer_profiling/
python -m pytest -q tests/test_customer_profiling.py
python -m streamlit run app.py   # Operator ▸ Customer Profiling, Consumer ▸ My Household / My DR Participation
```

## Inputs

| Input | Source |
|---|---|
| `aggregate_power_w` of the 50 households, resampled to 30 min | `households_1min_*.parquet` |
| `temperature_c`, `is_dr_event` | `grid_1min.csv` |
| `response` of each household to each past event | `dr_participation.csv` |
| Peak window (`peak_start`, `peak_end`) and deficit (kWh) | DR forecast (D+1); the historical replay uses `dr_events.csv` windows and the observed deficit |
| Control group of the event (optional) | Event manager |

## Outputs (`models/customer_profiling/kmeans_v1/`, accessed through `backend/services/segmentation_service.py`)

### Permanent (interface)

| Output | File | Columns | Screen |
|---|---|---|---|
| 1. Clustering + DR acceptance rate | `household_segments.csv` | `client_id`, 6 features, `cluster`, `segment`, `dr_priority`, `acceptance_rate`, `n_invitations`, `usual_evening_kwh`, `vs_similar_households_pct`, `comparison_message` | STEG ▸ Customer Profiling; Household ▸ My household |
| 2. Typical curves | `segment_load_profiles.csv` | `segment`, `season`, `day_type` (weekday/weekend), `slot` (0–47), `mean_kw_per_household` | STEG ▸ Customer Profiling; Household ▸ My household |

The acceptance rate is updated after every event. It is shown in the interface but is not a clustering feature.

### Before a DR event (orchestrator)

| Output | File | Columns | Used by |
|---|---|---|---|
| 3. Priority list | `event_targeting.csv` | `event_id`, `rank`, `client_id`, `segment`, `dr_priority`, `usual_window_kwh`, `acceptance_rate`, `expected_reduction_kwh`, `cumulative_expected_reduction_kwh`, `deficit_kwh`, `covers_deficit_at_rank` | Event manager |
| 4. Household context | `chatbot_context.csv` | `event_id`, `client_id`, `segment`, `usual_window_kwh`, `acceptance_rate`, `month_to_date_kwh`, `steg_band_millimes`, `projected_month_kwh`, `projected_band_millimes`, `usual_evening_kwh`, `vs_similar_households_pct`, `comparison_message` | Chatbot |

- The priority list is sorted by segment priority, then by usual consumption in the peak window. The cumulative expected reduction shows how far down the list the event manager must go to cover the deficit; the event manager decides who is notified and which households form the control group.
- For a new forecast window, `prepare_event(peak_start, peak_end, deficit_kwh)` returns outputs 3 and 4. It uses only data available at D−1 14:00.

### After a DR event: economics (pricing)

| Output | File | Columns | Rule |
|---|---|---|---|
| 5a. Cost of the event per household | `event_costs.csv` | `event_id`, `client_id`, `window_kwh`, `projected_month_kwh`, `projected_band_millimes`, `informed`, `economic_band`, `peak_price_millimes`, `event_cost_dt`, `surcharge_dt` | window kWh × peak price; peak price = band price × 1.5 (level 2). The control group and the economic band (≤ 100 kWh/month) are never surcharged |
| 5b. Savings per segment | `segment_savings.csv` | `event_id`, `segment`, `n_accepted`, `n_comparison`, `comparison_source`, `expected_kwh`, `actual_kwh`, `savings_kwh`, `reduction_pct` | see below |
| 5c. Rewards | `rewards.csv` | `event_id`, `client_id`, `segment`, `credited_kwh`, `segment_reduction_pct`, `reward_dt` | each accepting household gets an equal share of its segment's saved kWh, paid at the surcharge it avoided (band price × 50 %) |

**How 5b works.** The comparison group is the event's control group in that segment when there is one; otherwise it is the segment's non-accepting households. Then:
- expected = (comparison kWh in the window ÷ comparison kWh in the 4 h before) × accepters' kWh in the 4 h before;
- savings = expected − actual.

**STEG bands.** The whole month is billed at the band reached: ≤ 50 kWh → 62, ≤ 100 → 96, ≤ 200 → 176, ≤ 300 → 218, ≤ 500 → 341, above → 414 millimes/kWh.

## Method

1. **Clean.** Spikes above 15 kW are set to missing (46 in v3) and gaps of up to 2 h are interpolated. Data is then resampled to 30 min (mean kW, energy = kW × 0.5 h).
2. **Features (one row per household).** Computed on days without events, January to mid-June:

   | Feature | Definition |
   |---|---|
   | `annual_kwh` | annualised consumption |
   | `peak_kw` | 99th percentile of 30-min power |
   | `base_load_kw` | 5th percentile of 30-min power |
   | `load_factor` | mean / peak power |
   | `evening_share` | share of energy between 18:00 and 22:00 |
   | `cooling_kwh_per_deg` | slope of daily kWh vs daily mean °C on days above 24 °C |

3. **Cluster.** Standardised K-means, k = 4, 100 initialisations.
4. **Name the segments.** Each segment is named by its discriminating feature: lowest heat sensitivity → Low, no AC; highest base load → Daytime-occupied; highest evening energy → Large evening-peak; the rest → Medium AC users. The priority follows that order: 1 = Large evening-peak, 2 = Daytime-occupied, 3 = Medium AC, 4 = Low, no AC.
5. **Usual consumption.** "Usual" means the average over the last 28 days without events before the forecast time.

**Choice of the training window.** January–May contains only 11 days above 24 °C. On that window:
- silhouette 0.250, stability ARI 0.448;
- the "without AC" segment is 56 % AC owners.

As the data split plan allows, the training window was extended to January–mid-June (17 hot days):
- silhouette 0.252, ARI 0.491;
- the "without AC" segment is 50 % AC owners.

All 14 test events (June 17 → August) remain after the training window.

## Performance (test events June–September, `--evaluate`)

| Test | Result |
|---|---|
| Cluster quality (silhouette) | 0.25: the segments overlap; household consumption is a continuum |
| Stability (ARI, training window vs June–December) | 0.49: moderate; about half the households keep their segment once summer data is available |
| Meaningful segments | Large evening-peak: 100 % AC, 100 % electric water heater, 3.8 occupants. Low, no AC: lowest AC ownership (50 %), no daytime occupancy. Medium AC: only 18 % electric water heaters |
| Value for targeting | Large evening-peak = 36 % of homes, 43 % of peak consumption, **1.35 kWh saved per accepted event** vs 0.33 for Medium AC users (4×) |
| Economics (comparison-group savings) | Test events: 315 kWh estimated vs 245 kWh true (+29 %), per-event error 9.5 % of the no-DR consumption. Full year: +14 %, per-event error 11.9 %. Per segment on the test events: Low, no AC +3 %, Large evening-peak +22 %, Medium +33 %, Daytime-occupied +61 % |
| Priority list | Notifying only the top 50 % of the list captures **75 % of the true savings** (a random half would capture 50 %) |

## Limits

- **Few hot days in the training window.** The heat sensitivity of each home is estimated from 17 hot days, which explains the moderate stability. With a full summer of history, the segments would be retrained on a whole year.
- **Comparison-group savings overestimate summer savings.** Non-accepting homes are not a random sample, and with 50 homes each segment has only a few of them per event. A real control group set aside by the event manager (supported via `control_groups`) removes this bias.
- **Savings are reliable per segment, not per household.** Rewards are therefore shared at segment level.
- **Synthetic data.** Retrain and retest on real smart-meter data before deployment. `ground_truth_savings.csv` is read only by `backend/intelligence/evaluation.py`.
