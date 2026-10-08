from dataclasses import dataclass, field

STANDARD_COLUMNS = {
    "timestamp", "household_power_kw", "ac_power_kw", "water_heater_power_kw",
    "washing_machine_power_kw", "temperature_c", "dr_event", "dr_peak",
    "zone_demand_mw", "system_production_mw", "zone_pv_production_mw",
}


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    start: object | None
    end: object | None
    median_interval: object | None
    missing_intervals: int
    duplicate_timestamps: int
    missing_values: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
