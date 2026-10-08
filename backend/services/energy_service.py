import numpy as np
import pandas as pd

from backend.utils.time_utils import interval_hours


def integrate_power(data: pd.DataFrame, power_column: str) -> float:
    if data.empty or power_column not in data:
        return 0.0
    work = data[["timestamp", power_column]].copy()
    # Compute durations before excluding missing readings so gaps are never bridged.
    energy = work[power_column] * interval_hours(work["timestamp"])
    return float(energy.sum())


def energy_summary(data: pd.DataFrame, power_column: str = "household_power_kw") -> dict[str, float | int | bool]:
    values = data[power_column].dropna() if power_column in data else pd.Series(dtype=float)
    energy = integrate_power(data, power_column)
    peak = float(values.max()) if not values.empty else 0.0
    average = float(values.mean()) if not values.empty else 0.0
    total_samples = len(data)
    valid_samples = int(data[power_column].notna().sum()) if power_column in data else 0
    provenance = power_column.removesuffix("_kw") + "_was_missing"
    observed_samples = int((~data[provenance].astype(bool)).sum()) if provenance in data else valid_samples
    coverage = observed_samples / total_samples if total_samples else 0.0
    imputed_samples = int((data[provenance].astype(bool) & data[power_column].notna()).sum()) if provenance in data else 0
    return {"energy_kwh": energy, "peak_kw": peak, "average_kw": average,
            "load_factor": average / peak if peak > 0 else 0.0,
            "valid_samples": valid_samples, "observed_samples": observed_samples,
            "imputed_samples": imputed_samples, "total_samples": total_samples,
            "coverage": coverage, "energy_complete": coverage == 1.0}


def hourly_consumption(data: pd.DataFrame, power_column: str = "household_power_kw") -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "average_power_kw", "energy_kwh"])
    work = data[["timestamp", power_column]].dropna().set_index("timestamp")
    average = work[power_column].resample("1h").mean()
    # Integrate at native precision before grouping energy.
    native = data[["timestamp", power_column]].copy()
    native["energy_kwh"] = native[power_column] * interval_hours(native["timestamp"])
    energy = native.set_index("timestamp")["energy_kwh"].resample("1h").sum(min_count=1)
    return pd.DataFrame({"average_power_kw": average, "energy_kwh": energy}).reset_index()


def daily_consumption(data: pd.DataFrame, power_column: str = "household_power_kw") -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["date", "energy_kwh"])
    work = data[["timestamp", power_column]].copy()
    work["energy_kwh"] = work[power_column] * interval_hours(work["timestamp"])
    energy = work.assign(date=work.timestamp.dt.date).groupby("date", as_index=False)["energy_kwh"].sum(min_count=1)
    provenance = power_column.removesuffix("_kw") + "_was_missing"
    if provenance in data:
        coverage = data.assign(date=data.timestamp.dt.date, observed=~data[provenance].astype(bool)).groupby("date",as_index=False)["observed"].mean()
        energy = energy.merge(coverage,on="date",how="left").rename(columns={"observed":"observed_coverage"})
    else:
        energy["observed_coverage"] = data.groupby(data.timestamp.dt.date)[power_column].apply(lambda s:s.notna().mean()).to_numpy()
    return energy


def peak_offpeak_comparison(data: pd.DataFrame, peak_start: int = 17, peak_end: int = 22) -> dict[str, float]:
    if data.empty:
        return {"peak_average_kw": 0.0, "offpeak_average_kw": 0.0}
    mask = data.timestamp.dt.hour.between(peak_start, peak_end - 1)
    return {"peak_average_kw": float(data.loc[mask, "household_power_kw"].mean()),
            "offpeak_average_kw": float(data.loc[~mask, "household_power_kw"].mean())}
