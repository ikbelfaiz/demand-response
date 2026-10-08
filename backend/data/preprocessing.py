import pandas as pd

from backend.config import DatasetConfig
MW_FIELDS = ("zone_demand", "system_production", "zone_pv_production")
MAX_INTERPOLATION_MINUTES = 180


def _mapped(raw: pd.DataFrame, config: DatasetConfig, semantic: str) -> pd.Series | None:
    source = config.column_map.get(semantic)
    return raw[source].copy() if source and source in raw.columns else None


def prepare_dataset(raw: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Map source fields to stable names and clean analysis values without mutating raw."""
    out = pd.DataFrame(index=raw.index)
    ts = _mapped(raw, config, "timestamp")
    if ts is None:
        raise ValueError("Timestamp mapping is unavailable.")
    out["timestamp"] = pd.to_datetime(ts, errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("Invalid timestamps cannot be processed.")
    for semantic in ("household_power", *config.appliance_semantics):
        values = _mapped(raw, config, semantic)
        if values is not None:
            numeric = pd.to_numeric(values, errors="coerce")
            out[f"{semantic}_was_missing"] = numeric.isna()
            out[f"{semantic}_kw"] = numeric / 1000.0 if config.units.get(semantic) == "W" else numeric
    for semantic in MW_FIELDS:
        values = _mapped(raw, config, semantic)
        if values is not None:
            numeric = pd.to_numeric(values, errors="coerce")
            out[f"{semantic}_was_missing"] = numeric.isna()
            out[f"{semantic}_mw"] = numeric
    for semantic, target in (("temperature", "temperature_c"), ("dr_event", "dr_event"),
                             ("dr_peak", "dr_peak"), ("holiday", "is_holiday"),
                             ("ramadan", "is_ramadan")):
        values = _mapped(raw, config, semantic)
        if values is not None:
            numeric = pd.to_numeric(values, errors="coerce")
            if semantic == "temperature":
                out["temperature_was_missing"] = numeric.isna()
            out[target] = numeric
    for semantic in ("client_id", "region_id"):
        values = _mapped(raw, config, semantic)
        if values is not None:
            out[semantic] = values
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    sensor_cols = [c for c in out if c.endswith(("_kw", "_mw")) or c == "temperature_c"]
    # Only bounded gaps are repaired; long outages stay missing and provenance is retained.
    indexed = out.set_index("timestamp")
    for column in sensor_cols:
        original_missing = indexed[column].isna()
        groups = original_missing.ne(original_missing.shift()).cumsum()
        run_lengths = original_missing.groupby(groups).transform("sum")
        repaired = indexed[column].interpolate(method="time", limit_area="inside")
        indexed[column] = repaired.mask(original_missing & run_lengths.gt(MAX_INTERPOLATION_MINUTES))
    out = indexed.reset_index()
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    return out
