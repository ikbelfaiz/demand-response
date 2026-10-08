"""Leakage-safe half-hour feature and window construction for forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

FREQUENCY = "30min"
CONTEXT_SLOTS = 7 * 48
HORIZON_SLOTS = 48
TARGET_COLUMN = "aggregate_power_kw"


@dataclass(frozen=True)
class ForecastWindow:
    client_id: str
    forecast_date: pd.Timestamp
    context_timestamps: pd.DatetimeIndex
    context_kw: np.ndarray
    target_timestamps: pd.DatetimeIndex
    target_kw: np.ndarray | None


def expected_timestamps(forecast_date: object) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    day = pd.Timestamp(forecast_date).normalize()
    context = pd.date_range(day - pd.DateOffset(days=7), periods=CONTEXT_SLOTS, freq=FREQUENCY)
    target = pd.date_range(day, periods=HORIZON_SLOTS, freq=FREQUENCY)
    return context, target


def build_window(frame: pd.DataFrame, client_id: str, forecast_date: object,
                 *, require_target: bool = False) -> ForecastWindow:
    """Build an exact 336-to-48 window; never interpolate or use future context."""
    if not {"timestamp", TARGET_COLUMN}.issubset(frame):
        raise ValueError(f"Frame must contain timestamp and {TARGET_COLUMN}.")
    context_index, target_index = expected_timestamps(forecast_date)
    values = (frame[["timestamp", TARGET_COLUMN]].copy()
              .assign(timestamp=lambda x: pd.to_datetime(x.timestamp))
              .drop_duplicates("timestamp", keep=False)
              .set_index("timestamp")[TARGET_COLUMN]
              .sort_index())
    context = values.reindex(context_index)
    if context.isna().any():
        missing = context.index[context.isna()]
        raise ValueError(f"Forecast context is incomplete ({len(missing)} missing slots; first {missing[0]}).")
    target = values.reindex(target_index)
    target_values: np.ndarray | None = None
    if target.notna().all():
        target_values = target.to_numpy(dtype=np.float32)
    elif require_target:
        raise ValueError("Forecast target is incomplete and cannot be used for evaluation.")
    return ForecastWindow(
        client_id=client_id,
        forecast_date=pd.Timestamp(forecast_date).normalize(),
        context_timestamps=context_index,
        context_kw=context.to_numpy(dtype=np.float32),
        target_timestamps=target_index,
        target_kw=target_values,
    )


def eligible_forecast_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return dates having a complete seven-day context and complete target day."""
    if frame.empty:
        return pd.DatetimeIndex([])
    first = pd.Timestamp(frame.timestamp.min()).normalize() + pd.DateOffset(days=7)
    last = pd.Timestamp(frame.timestamp.max()).normalize()
    dates = []
    for day in pd.date_range(first, last, freq="1D"):
        try:
            build_window(frame, "candidate", day, require_target=True)
            dates.append(day)
        except ValueError:
            continue
    return pd.DatetimeIndex(dates)


def chronological_split(dates: Iterable[object], train_end: object, validation_end: object
                        ) -> dict[str, pd.DatetimeIndex]:
    """Split by target day, ensuring a target day belongs to exactly one partition."""
    index = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().sort_values().unique()
    train_boundary = pd.Timestamp(train_end).normalize()
    validation_boundary = pd.Timestamp(validation_end).normalize()
    if validation_boundary <= train_boundary:
        raise ValueError("validation_end must follow train_end")
    return {
        "train": index[index <= train_boundary],
        "validation": index[(index > train_boundary) & (index <= validation_boundary)],
        "test": index[index > validation_boundary],
    }


def seasonal_persistence(context_kw: np.ndarray) -> np.ndarray:
    values = np.asarray(context_kw, dtype=np.float32)
    if values.shape != (CONTEXT_SLOTS,):
        raise ValueError(f"Expected {CONTEXT_SLOTS} context values, got {values.shape}.")
    return values[:HORIZON_SLOTS].copy()


def normalize_context(context_kw: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Notebook-compatible per-origin scaling fitted only to the historical context."""
    values = np.asarray(context_kw, dtype=np.float32)
    if values.shape != (CONTEXT_SLOTS,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {CONTEXT_SLOTS} finite context values.")
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if std < 1e-8:
        raise ValueError("Context variance is too small for standardization.")
    return (values - mean) / (std + 1e-8), mean, std
