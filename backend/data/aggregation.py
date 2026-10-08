"""Canonical one-minute to operational half-hour energy conversion."""
from __future__ import annotations

import pandas as pd

from backend.config import OPERATIONAL_FREQUENCY


def resample_power(frame: pd.DataFrame, power_columns: list[str], *, timestamp: str = "timestamp",
                   frequency: str = OPERATIONAL_FREQUENCY) -> pd.DataFrame:
    """Average power and derive interval energy; power is never summed."""
    if frame.empty: return pd.DataFrame(columns=[timestamp])
    indexed=frame.set_index(timestamp).sort_index()
    result=pd.DataFrame(index=indexed.resample(frequency).size().index)
    interval_hours=pd.tseries.frequencies.to_offset(frequency).nanos/3_600_000_000_000
    for column in power_columns:
        if column not in indexed: continue
        result[column]=pd.to_numeric(indexed[column],errors="coerce").resample(frequency).mean()
        result[f"{column.removesuffix('_kw')}_energy_kwh"]=result[column]*interval_hours
    return result.reset_index(names=timestamp)


def operational_grid(raw: pd.DataFrame) -> pd.DataFrame:
    powers=["households_consumption_kw","steg_production_kw","pv_production_kw"]
    out=resample_power(raw,powers); indexed=raw.set_index("timestamp")
    for column,agg in (("temperature_c","mean"),("is_holiday","max"),("is_ramadan","max"),("is_dr_event","max"),("is_dr_peak","max")):
        out[column]=getattr(indexed[column].resample(OPERATIONAL_FREQUENCY),agg)().to_numpy()
    out["available_supply_kw"]=out.steg_production_kw+out.pv_production_kw
    out["margin_kw"]=out.available_supply_kw-out.households_consumption_kw
    return out


def operational_household(raw: pd.DataFrame) -> pd.DataFrame:
    work=raw.copy(); columns=[]
    for source in ["aggregate_power_w","ac_power_w","water_heater_power_w","washing_machine_power_w"]:
        if source not in work:
            continue
        target=source.replace("_w","_kw"); work[target]=pd.to_numeric(work[source],errors="coerce")/1000; columns.append(target)
    return resample_power(work,columns)
