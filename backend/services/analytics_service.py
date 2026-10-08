import pandas as pd


def filter_dates(data: pd.DataFrame, start, end=None) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    start_ts = pd.Timestamp(start)
    requested_end = pd.Timestamp(end if end is not None else start)
    if pd.isna(start_ts) or pd.isna(requested_end):
        raise ValueError("Start and end dates must be valid dates.")
    end_ts = requested_end + pd.offsets.Day(1)
    return data.loc[(data.timestamp >= start_ts) & (data.timestamp < end_ts)].copy()


def chart_series(data: pd.DataFrame, frequency: str = "5min") -> pd.DataFrame:
    """Downsample power with means and event flags with maxima; analytics stays native."""
    if data.empty:
        return data.copy()
    aggregations = {}
    for col in data.select_dtypes("number"):
        aggregations[col] = "max" if col in {"dr_event", "dr_peak"} else "mean"
    return data.set_index("timestamp").resample(frequency).agg(aggregations).reset_index()


def display_frequency(data: pd.DataFrame) -> str:
    if data.empty:
        return "5min"
    days = max((data.timestamp.max() - data.timestamp.min()).total_seconds() / 86400, 0)
    if days <= 1:
        return "5min"
    if days <= 7:
        return "30min"
    if days <= 31:
        return "2h"
    return "1D"
