from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    column_map: dict[str, str]
    units: dict[str, str] = field(default_factory=dict)
    timezone: str | None = None

    @property
    def appliance_semantics(self) -> tuple[str, ...]:
        return tuple(key for key in self.column_map if key.endswith("_power") and key != "household_power")


DATASETS = {
    "dr_2025": DatasetConfig(
        name="2025 Tunisia household and zone DR dataset",
        path=DATA_DIR / "dr_dataset_2025_1min_v2.csv",
        column_map={
            "timestamp": "timestamp",
            "household_power": "aggregate_power_w",
            "ac_power": "ac_power_w",
            "water_heater_power": "water_heater_power_w",
            "washing_machine_power": "washing_machine_power_w",
            "temperature": "temperature_c",
            "dr_event": "is_dr_event",
            "dr_peak": "is_dr_peak",
            "zone_demand": "zone_consumption_mw",
            "system_production": "steg_production_mw",
            "zone_pv_production": "pv_production_mw",
            "client_id": "client_id",
            "region_id": "region_id",
            "holiday": "is_holiday",
            "ramadan": "is_ramadan",
        },
        units={
            "household_power": "W", "ac_power": "W", "water_heater_power": "W",
            "washing_machine_power": "W", "temperature": "degC",
            "zone_demand": "MW", "system_production": "MW", "zone_pv_production": "MW",
        },
    )
}


def get_dataset_config(name: str = "dr_2025") -> DatasetConfig:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Registered datasets: {', '.join(DATASETS)}")
    return DATASETS[name]
