"""Customer profiling adapter — builds the priority list the event manager consumes.

Output columns (one row per household):
    rank, client_id, segment, usual_window_kwh, month_to_date_kwh, projected_month_kwh,
    contracted_kva, acceptance_rate, expected_reduction_kwh, covers_deficit_at_rank, comparison_message
Sorted by segment priority, then by usual consumption in the event window (descending).

Two sources, chosen by default_targeting():
  CustomerProfilingTargeting  the team's K-means model (backend/intelligence/customer_profiling.py,
                              artifact models/customer_profiling/kmeans_v1) through
                              segmentation_service.prepare_event(peak_start, peak_end, deficit_kwh, as_of)
  LocalTargeting              fallback when the profiling outputs are not installed: same columns,
                              computed here from the 30-min meter readings (segments "Unknown").
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


PRIORITY_COLUMNS = ["rank", "client_id", "segment", "usual_window_kwh", "month_to_date_kwh", "projected_month_kwh",
                    "contracted_kva", "acceptance_rate", "expected_reduction_kwh", "covers_deficit_at_rank",
                    "comparison_message"]


class LocalTargeting:
    source = "local priority list"

    def priority_list(self, meter30, households, start, end, asof, exclude_days, deficit_kwh=None) -> pd.DataFrame:
        df = priority_list(meter30, households, start, end, asof, exclude_days)
        df["expected_reduction_kwh"] = np.nan
        df["covers_deficit_at_rank"] = False
        df["comparison_message"] = None
        return df[PRIORITY_COLUMNS]


class CustomerProfilingTargeting:
    """Priority list + household context from the team's customer profiling model.

    Only data available at `asof` (the decision time, D-1) is read by prepare_event.
    """

    def __init__(self, prepare_event=None, version: str | None = None):
        from backend.services import segmentation_service as ss
        self._prepare = prepare_event or ss.prepare_event
        if version is None:
            meta = ss.load_metadata()
            version = f"{meta.get('model_name', 'customer_profiling')}_{meta.get('model_version', '')}".rstrip("_")
        self.source = version

    def priority_list(self, meter30, households, start, end, asof, exclude_days, deficit_kwh=None) -> pd.DataFrame:
        targeting, context = self._prepare(start, end, float(deficit_kwh or 0.0), as_of=asof)
        ctx = context.set_index("client_id")
        df = targeting[["rank", "client_id", "segment", "usual_window_kwh", "acceptance_rate",
                        "expected_reduction_kwh", "covers_deficit_at_rank"]].copy()
        df = df[df.client_id.isin(households.index)]
        df["month_to_date_kwh"] = df.client_id.map(ctx["month_to_date_kwh"]).round(1)
        df["projected_month_kwh"] = df.client_id.map(ctx["projected_month_kwh"]).round(1)
        df["comparison_message"] = df.client_id.map(ctx["comparison_message"]) if "comparison_message" in ctx else None
        df["contracted_kva"] = df.client_id.map(households["contracted_kva"])
        # households the profiling model does not know (new contracts) go to the end of the list
        missing = households.index.difference(df.client_id)
        if len(missing):
            extra = LocalTargeting().priority_list(meter30, households.loc[missing], start, end, asof, exclude_days)
            df = pd.concat([df, extra], ignore_index=True)
        df["usual_window_kwh"] = df["usual_window_kwh"].fillna(0).round(3)
        df["acceptance_rate"] = df["acceptance_rate"].fillna(households["acceptance_rate"].mean() if len(households) else 0.5)
        fallback_proj = month_projection(meter30, asof)[1]
        df["projected_month_kwh"] = df["projected_month_kwh"].fillna(df.client_id.map(fallback_proj)).fillna(0)
        df["month_to_date_kwh"] = df["month_to_date_kwh"].fillna(0)
        df["covers_deficit_at_rank"] = df["covers_deficit_at_rank"].fillna(False).astype(bool)
        df = df.reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df[PRIORITY_COLUMNS]


def profiling_installed() -> bool:
    try:
        from backend.services.segmentation_service import profiling_available
    except ImportError:
        return False
    return profiling_available()


def default_targeting():
    """The team's customer profiling model when its outputs are installed, otherwise the local list."""
    return CustomerProfilingTargeting() if profiling_installed() else LocalTargeting()
