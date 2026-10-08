import pandas as pd

from backend.config import DatasetConfig
from .schema import ValidationReport


def validate_dataset(raw: pd.DataFrame, config: DatasetConfig) -> ValidationReport:
    timestamp_source = config.column_map.get("timestamp")
    if not timestamp_source or timestamp_source not in raw.columns:
        raise ValueError("The configured timestamp column is missing.")
    if "household_power" not in config.column_map or config.column_map["household_power"] not in raw.columns:
        raise ValueError("A mapped household consumption power column is required.")
    parsed = pd.to_datetime(raw[timestamp_source], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{int(parsed.isna().sum())} timestamps could not be parsed.")
    ordered = parsed.sort_values()
    deltas = ordered.diff().dropna()
    median = deltas.median() if not deltas.empty else None
    missing_intervals = 0
    if median is not None and median > pd.Timedelta(0):
        missing_intervals = int(((deltas / median) - 1).clip(lower=0).round().sum())
    warnings = []
    if parsed.dt.tz is None:
        warnings.append("Timestamps have no timezone metadata; local civil time is retained without localization.")
    missing = {c: int(v) for c, v in raw.isna().sum().items() if v}
    return ValidationReport(
        row_count=len(raw), start=parsed.min() if len(raw) else None,
        end=parsed.max() if len(raw) else None, median_interval=median,
        missing_intervals=missing_intervals,
        duplicate_timestamps=int(parsed.duplicated().sum()), missing_values=missing,
        warnings=tuple(warnings),
    )

