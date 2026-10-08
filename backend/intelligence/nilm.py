"""NILM dataset and inference interfaces; no fabricated estimates."""
import pandas as pd

TARGET_COLUMNS = ["ac_power_kw", "water_heater_power_kw", "washing_machine_power_kw"]


def build_training_frame(panel_operational: pd.DataFrame) -> pd.DataFrame:
    required = {"client_id", "timestamp", "aggregate_power_kw", *TARGET_COLUMNS}
    missing = required.difference(panel_operational.columns)
    if missing: raise ValueError(f"Missing NILM training fields: {sorted(missing)}")
    return panel_operational.dropna(subset=["aggregate_power_kw", *TARGET_COLUMNS]).copy()


def household_split(frame: pd.DataFrame, test_households: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = frame.client_id.isin(test_households)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def estimate_appliances(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError("NILM is not trained; measured submeters are shown only where available.")
