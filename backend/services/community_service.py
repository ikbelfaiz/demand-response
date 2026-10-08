"""Community metrics computed from the modeled feeder measurements."""
import pandas as pd


def summary(frame: pd.DataFrame) -> dict[str, float]:
    demand = frame.get("households_consumption_kw", pd.Series(dtype=float))
    return {
        "energy_kwh": float(frame.get("households_consumption_energy_kwh", pd.Series(dtype=float)).sum(min_count=1)),
        "average_demand_kw": float(demand.mean()), "maximum_demand_kw": float(demand.max()),
        "steg_energy_kwh": float(frame.get("steg_production_energy_kwh", pd.Series(dtype=float)).sum(min_count=1)),
        "pv_energy_kwh": float(frame.get("pv_production_energy_kwh", pd.Series(dtype=float)).sum(min_count=1)),
        "average_margin_kw": float(frame.get("margin_kw", pd.Series(dtype=float)).mean()),
    }


def demand_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.assign(date=frame.timestamp.dt.date, hour=frame.timestamp.dt.hour)
    return work.pivot_table(index="date", columns="hour", values="households_consumption_kw", aggfunc="mean")


def peak_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.assign(hour=frame.timestamp.dt.hour, month=frame.timestamp.dt.month)
    return work.groupby("hour", as_index=False).agg(peak_intervals=("is_dr_peak", "sum"), observations=("is_dr_peak", "count"))
