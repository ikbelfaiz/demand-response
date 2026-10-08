import numpy as np
import pandas as pd

from backend.services.energy_service import daily_consumption, integrate_power
from backend.utils.time_utils import interval_hours


APPLIANCE_LABELS = {
    "ac_power_kw": "Air conditioning",
    "water_heater_power_kw": "Water heater",
    "washing_machine_power_kw": "Washing machine",
}


def appliance_columns(data: pd.DataFrame) -> list[str]:
    """Discover configured standardized appliance power channels."""
    return [column for column in data if column.endswith("_power_kw") and column != "household_power_kw" and data[column].notna().any()]


def appliance_label(column: str) -> str:
    return APPLIANCE_LABELS.get(column, column.removesuffix("_power_kw").replace("_", " ").title())


def appliance_breakdown(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in appliance_columns(data):
        rows.append({"appliance": appliance_label(column), "energy_kwh": integrate_power(data, column),
                     "coverage": float(data[column].notna().mean()) if len(data) else 0.0})
    return pd.DataFrame(rows, columns=["appliance", "energy_kwh", "coverage"])


def hourly_profile(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["hour", "average_power_kw", "p95_power_kw"])
    grouped = data.groupby(data.timestamp.dt.hour)["household_power_kw"]
    return pd.DataFrame({"average_power_kw": grouped.mean(), "p95_power_kw": grouped.quantile(.95)}).rename_axis("hour").reset_index()


def hourly_heatmap(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    work = data.assign(date=data.timestamp.dt.date, hour=data.timestamp.dt.hour)
    return work.pivot_table(index="date", columns="hour", values="household_power_kw", aggfunc="mean")


def peak_periods(data: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    columns = ["timestamp", "household_power_kw"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    return data[columns].dropna().nlargest(count, "household_power_kw").reset_index(drop=True)


def household_distribution(data: pd.DataFrame) -> pd.Series:
    return data.get("household_power_kw", pd.Series(dtype=float)).dropna()


def daily_trend(data: pd.DataFrame) -> pd.DataFrame:
    return daily_consumption(data)
