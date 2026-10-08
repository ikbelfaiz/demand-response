import pandas as pd


REGIONAL_COLUMNS = ["zone_demand_mw", "system_production_mw", "zone_pv_production_mw"]


def regional_summary(data: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {"valid_samples": len(data)}
    for column in REGIONAL_COLUMNS:
        values = data[column].dropna() if column in data else pd.Series(dtype=float)
        result[f"{column}_average"] = float(values.mean()) if not values.empty else 0.0
        result[f"{column}_peak"] = float(values.max()) if not values.empty else 0.0
        provenance = column.removesuffix("_mw") + "_was_missing"
        result[f"{column}_coverage"] = float((~data[provenance].astype(bool)).mean()) if len(data) and provenance in data else (float(values.size / len(data)) if len(data) else 0.0)
    return result


def regional_timeseries(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["timestamp", *REGIONAL_COLUMNS])
    available = [column for column in REGIONAL_COLUMNS if column in data]
    return data.set_index("timestamp")[available].resample(frequency).mean().reset_index()


def regional_daily_trends(data: pd.DataFrame) -> pd.DataFrame:
    return regional_timeseries(data, "1D")


def regional_peak_periods(data: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    columns = ["timestamp", "zone_demand_mw"]
    if data.empty or "zone_demand_mw" not in data:
        return pd.DataFrame(columns=columns)
    return data[columns].dropna().nlargest(count, "zone_demand_mw").reset_index(drop=True)


def reported_supply_comparison(data: pd.DataFrame, frequency: str = "1h") -> pd.DataFrame:
    """Compare only supplied production fields; this is not a complete grid balance."""
    frame = regional_timeseries(data, frequency)
    if not frame.empty and {"system_production_mw", "zone_pv_production_mw", "zone_demand_mw"}.issubset(frame):
        frame["reported_production_mw"] = frame.system_production_mw + frame.zone_pv_production_mw
        frame["reported_production_minus_demand_mw"] = frame.reported_production_mw - frame.zone_demand_mw
    return frame


def reported_comparison_summary(data: pd.DataFrame) -> dict[str, float | None]:
    comparison = reported_supply_comparison(data, "1h")
    column = "reported_production_minus_demand_mw"
    return {"mean_reported_production_minus_demand_mw":
            float(comparison[column].mean()) if column in comparison and comparison[column].notna().any() else None}
