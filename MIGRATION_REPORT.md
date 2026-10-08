# Demand Response v3 migration report

## Outcome

The runtime uses one canonical complete v3 dataset. Navigation provides Operator and Consumer perspectives. A saved Energy-TTM household forecaster is exposed without fabricating future actuals; automatic DR decisions remain disabled.

## Verified dataset

| Finding | Value |
|---|---:|
| Grid rows | 525,600 |
| Household rows | 26,280,000 |
| Household IDs | C001–C050 |
| Submetered homes | 10 |
| Historical events | 36 |
| Participation rows | 1,800 |
| Responses | 975 accept, 515 decline, 310 no response |
| Duplicate measurement keys | 0 |

Canonical completion corrected 2,280 temperature, 214 feeder-demand, 315 PV, 106,025 aggregate household, 13,252 AC, 11,675 water-heater, and 16,230 washing-machine readings. The generator rule reconstructed 528 absent peak labels. Existing event, peak, and calendar labels were preserved. No boundary fallback was required.

Pre-completion files and checksums are stored in `data/original_v3_backup/`. Appliance nulls remain only for the 40 homes without submeters.

## Generator interpretation

Available supply is derived from `STEG capacity - 0.35 × zone demand + PV` and calibrated to neighborhood scale. Feeder demand already includes the modeled 3% loss. Historical event decisions used an imperfect synthetic prior-day forecast. These assumptions do not describe verified real-world shortages.

## Delivered architecture and pages

Cached canonical loaders use CSV and Parquet pushdown. Operational aggregation produces mean kW and kWh at 30-minute resolution. Operator pages cover community, grid, events, and households; the Data Explorer page and navigation were removed while backend validation stayed intact. Consumer My Energy includes saved-model historical evaluation and January 1, 2026 next-day forecasting.

## Future work and limitations

Community forecasting, NILM inference, baseline estimation, flexibility modeling, automatic activation, tariffs, and real notifications remain future work. The household forecast alone does not trigger DR. Timestamps lack timezone metadata. Portfolio scans can take several seconds on first use. Scenario surcharge levels have no monetary interpretation.

## Forecast model migration

The notebook-selected Energy-TTM architecture was adapted from 168 hourly inputs/24 outputs to 336 half-hour inputs/48 outputs by preserving seven daily patches (patch length/stride 24 → 48) and retraining the resized patcher and forecast head with the mixer core. Five CPU epochs used 13,300 training windows and 1,550 validation windows; best validation loss was 0.652229. On 3,050 held-out household-days, Energy-TTM achieved mean MAE 0.2472 kW and RMSE 0.3865 kW versus seasonal persistence at 0.2787 kW and 0.5052 kW. Peak timing remained much worse (367.2 versus 22.8 minutes), a material limitation documented in `FORECASTING_MODEL.md`.
