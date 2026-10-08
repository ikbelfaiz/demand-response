import numpy as np
import pandas as pd


def interval_hours(timestamps: pd.Series) -> pd.Series:
    """Forward sample durations, using the validated median for the final sample."""
    ts = pd.to_datetime(timestamps)
    if ts.empty:
        return pd.Series(dtype=float, index=timestamps.index)
    delta = ts.shift(-1).sub(ts).dt.total_seconds().div(3600)
    valid = delta[(delta > 0) & np.isfinite(delta)]
    fallback = float(valid.median()) if not valid.empty else 0.0
    return delta.where(delta > 0, fallback).fillna(fallback).clip(lower=0)


def event_windows(data: pd.DataFrame, flag: str = "dr_event") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if data.empty or flag not in data:
        return []
    active = data[flag].fillna(0).eq(1)
    groups = active.ne(active.shift()).cumsum()
    windows = []
    for _, part in data.loc[active].groupby(groups[active]):
        step = pd.to_timedelta(interval_hours(part["timestamp"]).median(), unit="h")
        windows.append((part["timestamp"].iloc[0], part["timestamp"].iloc[-1] + step))
    return windows

