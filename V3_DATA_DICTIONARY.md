# V3 data dictionary

The synthetic dataset covers 2025 at one-minute resolution. Timestamps are timezone-naive local civil time. Runtime loaders use the canonical source filenames directly. Installed numerical sensors are complete; non-submetered appliance channels remain structurally unavailable.

## Relationships

`households_info.client_id` is referenced by meter readings and participation. `dr_events.event_id` is referenced by participation and research-only savings. There are 50 households, 36 events, and 1,800 complete event-household response pairs.

## `grid_1min.csv`

| Field | Unit / meaning |
|---|---|
| `timestamp` | One-minute timestamp |
| `region_id` | Synthetic region `TUN` |
| `temperature_c` | Synthetic temperature, °C |
| `is_holiday`, `is_ramadan` | Calendar flags |
| `households_consumption_kw` | Feeder power, kW; includes modeled 3% loss and small noise |
| `steg_production_kw` | Calibrated available STEG allocation, kW |
| `pv_production_kw` | Calibrated available PV allocation, kW |
| `is_dr_event` | Historical synthetic event flag |
| `is_dr_peak` | Centered-15-minute demand-over-supply flag; previously absent labels were deterministically reconstructed with the generator rule |

## Household parquet files

Columns are `timestamp`, `client_id`, `aggregate_power_w`, `ac_power_w`, `water_heater_power_w`, and `washing_machine_power_w`. Aggregate readings are complete for all 50 homes. Appliance channels are complete for the 10 panel homes and null for all timestamps in the 40 non-panel homes.

## Household profiles

`households_info.csv` contains region, occupants, appliance ownership, water-heater rating, daytime occupancy, and submeter availability. A null heater rating means the home has no electric water heater.

## Events and research truth

`dr_events.csv` contains synthetic event times, scenario levels, and reduction targets. `dr_participation.csv` contains `accept`, `decline`, or `no_response`. `ground_truth_savings.csv` contains synthetic natural counterfactual and actual event energy and is available only to research evaluation.

## Operational derivation

Half-hour power is the arithmetic mean of 30 one-minute readings. Energy is `mean power (kW) × 0.5 h`. Aggregate and installed-submeter channels produce 48 populated intervals per day.
