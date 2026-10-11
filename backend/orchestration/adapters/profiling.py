"""Customer profiling adapter — builds the priority list the event manager consumes.

Output columns (one row per household):
    rank, client_id, segment, usual_window_kwh, month_to_date_kwh, projected_month_kwh,
    contracted_kva, acceptance_rate
Sorted by segment priority, then by usual consumption in the event window (descending).
Segments come from data/customer_profiles.csv (K-means customer profiling); the consumption columns
are computed here from the 30-min smart-meter readings. Without that file every household is
"Unknown" and the list is sorted by consumption only.
"""
import calendar

import numpy as np
import pandas as pd

from backend.orchestration import settings as config

SEGMENT_PRIORITY = {
    "Large evening-peak households": 1,
    "Daytime-occupied, high base load": 2,
    "Medium AC users": 3,
    "Low consumers without AC": 4,
}


def usual_window_kwh(meter30: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                     asof: pd.Timestamp, exclude_days: set) -> pd.Series:
    """Mean kWh in the same clock window over the last N days before `asof`, event days excluded."""
    sums = []
    d = start.normalize() - pd.Timedelta(days=1)
    while len(sums) < config.USUAL_WINDOW_LOOKBACK_DAYS and d > start - pd.Timedelta(days=60):
        if d.date() not in exclude_days and d + (end - start.normalize()) <= asof:
            s = d + (start - start.normalize())
            e = d + (end - start.normalize())
            sums.append(meter30.loc[s:e - pd.Timedelta(seconds=1)].sum(min_count=1))
        d -= pd.Timedelta(days=1)
    if not sums:
        return pd.Series(np.nan, index=meter30.columns)
    return pd.concat(sums, axis=1).mean(axis=1)


def month_projection(meter30: pd.DataFrame, asof: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Month-to-date kWh and projected end-of-month kWh (gaps filled with the household mean)."""
    m0 = asof.replace(day=1).normalize()
    mtd_df = meter30.loc[m0:asof - pd.Timedelta(seconds=1)]
    days_in_month = calendar.monthrange(asof.year, asof.month)[1]
    elapsed_days = (asof - m0) / pd.Timedelta(days=1)
    if elapsed_days < 3:   # too early in the month: use the previous 30 days
        prev = meter30.loc[asof - pd.Timedelta(days=30):asof - pd.Timedelta(seconds=1)]
        rate = prev.mean() * 48
        return mtd_df.sum(), rate * days_in_month
    mtd = mtd_df.sum()
    rate = mtd_df.mean() * 48                        # kWh/day, robust to missing slots
    return mtd, mtd + rate * (days_in_month - elapsed_days)


def priority_list(meter30: pd.DataFrame, households: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                  asof: pd.Timestamp, exclude_days: set) -> pd.DataFrame:
    usual = usual_window_kwh(meter30, start, end, asof, exclude_days)
    mtd, proj = month_projection(meter30, asof)
    df = pd.DataFrame({
        "client_id": households.index,
        "segment": households["segment"].to_numpy(),
        "usual_window_kwh": usual.reindex(households.index).fillna(0).round(3).to_numpy(),
        "month_to_date_kwh": mtd.reindex(households.index).round(1).to_numpy(),
        "projected_month_kwh": proj.reindex(households.index).round(1).to_numpy(),
        "contracted_kva": households["contracted_kva"].to_numpy(),
        "acceptance_rate": households["acceptance_rate"].to_numpy(),
    })
    df["segment_priority"] = df["segment"].map(SEGMENT_PRIORITY).fillna(9).astype(int)
    df = df.sort_values(["segment_priority", "usual_window_kwh"], ascending=[True, False]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
