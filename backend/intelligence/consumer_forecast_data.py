"""Leakage-safe household data preparation for consumer D+1 forecasting only."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROJECT_ROOT
from .forecast_features import CONTEXT_SLOTS, HORIZON_SLOTS, ForecastWindow, expected_timestamps

MIN_SOURCE_COVERAGE = 0.8
ORIGINAL_DATA = PROJECT_ROOT / "data" / "original_v3_backup"


def original_household_path(client_id: str) -> Path:
    number = int(client_id.removeprefix("C"))
    if not 1 <= number <= 50:
        raise KeyError(f"Unknown household {client_id!r}")
    name = ("households_1min_C001_C025.parquet" if number <= 25 else
            "households_1min_C026_C050.parquet")
    return ORIGINAL_DATA / name


@lru_cache(maxsize=50)
def load_causal_household(client_id: str) -> pd.DataFrame:
    """Aggregate original source data and fill contexts using past values only.

    A 30-minute measurement is valid with at least 24 of 30 source minutes.
    Remaining gaps are forward-filled for model context only.  The untouched
    observed series remains available for targets and evaluation.
    """
    raw = pd.read_parquet(
        original_household_path(client_id),
        columns=["timestamp", "client_id", "aggregate_power_w"],
        filters=[("client_id", "==", client_id)], engine="pyarrow",
    )
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    values = pd.to_numeric(raw.aggregate_power_w, errors="coerce") / 1000.0
    grouped = values.groupby(raw.timestamp.dt.floor("30min"))
    mean_kw, count = grouped.mean(), grouped.count()
    index = pd.date_range(raw.timestamp.min().floor("30min"),
                          raw.timestamp.max().floor("30min"), freq="30min")
    coverage = (count / 30.0).reindex(index, fill_value=0.0)
    observed = mean_kw.reindex(index).where(coverage >= MIN_SOURCE_COVERAGE)
    causal = observed.ffill()
    return pd.DataFrame({
        "timestamp": index,
        "aggregate_power_kw": observed.to_numpy(dtype=float),
        "context_power_kw": causal.to_numpy(dtype=float),
        "source_coverage": coverage.to_numpy(dtype=float),
        "context_imputed": observed.isna().to_numpy(),
    })


def build_consumer_window(frame: pd.DataFrame, client_id: str, forecast_date: object,
                          *, require_target: bool = False) -> ForecastWindow:
    """Build 336→48 data where context imputation is causal and targets stay raw."""
    context_index, target_index = expected_timestamps(forecast_date)
    indexed = frame.drop_duplicates("timestamp", keep=False).set_index("timestamp").sort_index()
    context = indexed.context_power_kw.reindex(context_index)
    if context.isna().any():
        raise ValueError("Consumer forecast context is incomplete after causal preparation.")
    target = indexed.aggregate_power_kw.reindex(target_index)
    target_values = target.to_numpy(dtype=np.float32) if target.notna().all() else None
    if require_target and target_values is None:
        raise ValueError("Consumer forecast target contains an interval below 80% source coverage.")
    return ForecastWindow(
        client_id=client_id,
        forecast_date=pd.Timestamp(forecast_date).normalize(),
        context_timestamps=context_index,
        context_kw=context.to_numpy(dtype=np.float32),
        target_timestamps=target_index,
        target_kw=target_values,
    )


def consumer_baselines(frame: pd.DataFrame, forecast_date: object) -> dict[str, np.ndarray]:
    """Causal household baselines using only information before forecast origin."""
    day = pd.Timestamp(forecast_date).normalize()
    indexed = frame.set_index("timestamp").sort_index()
    history = indexed.loc[indexed.index < day, "context_power_kw"]
    target_slots = np.arange(HORIZON_SLOTS)

    def profile_for(date: pd.Timestamp) -> np.ndarray:
        index = pd.date_range(date, periods=HORIZON_SLOTS, freq="30min")
        return history.reindex(index).to_numpy(dtype=np.float32)

    day_profiles, day_types = [], []
    for prior_day in pd.date_range(history.index.min().normalize(), day - pd.DateOffset(days=1), freq="D"):
        profile = profile_for(prior_day)
        if np.isfinite(profile).all():
            day_profiles.append(profile)
            day_types.append(prior_day.dayofweek >= 5)
    profiles = np.asarray(day_profiles, dtype=np.float32)
    if profiles.size == 0:
        raise ValueError("No causal history is available for baselines.")
    same_type = profiles[np.asarray(day_types) == (day.dayofweek >= 5)]
    if same_type.size == 0:
        same_type = profiles
    return {
        "Previous-day persistence": profile_for(day - pd.DateOffset(days=1)),
        "Previous-week persistence": profile_for(day - pd.DateOffset(days=7)),
        "Historical slot median": np.median(profiles[:, target_slots], axis=0).astype(np.float32),
        "Weekday/weekend profile": np.median(same_type[:, target_slots], axis=0).astype(np.float32),
    }


def target_coverage(frame: pd.DataFrame, forecast_date: object) -> np.ndarray:
    _, target_index = expected_timestamps(forecast_date)
    return (frame.set_index("timestamp").source_coverage.reindex(target_index)
            .to_numpy(dtype=np.float32))
