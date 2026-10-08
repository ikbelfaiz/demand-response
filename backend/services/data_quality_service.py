import pandas as pd


def _provenance_flag(column: str) -> str:
    if column == "temperature_c":
        return "temperature_was_missing"
    return f"{column.removesuffix('_kw').removesuffix('_mw')}_was_missing"


def missing_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measurement_columns = [c for c in data if c.endswith(("_kw", "_mw")) or c in {"temperature_c", "dr_peak"}]
    for column in measurement_columns:
        source_flag = _provenance_flag(column)
        original_missing = data[source_flag] if source_flag in data else data[column].isna()
        rows.append({"channel": column, "original_missing": int(original_missing.sum()),
                     "remaining_missing": int(data[column].isna().sum()),
                     "valid_percent": float(data[column].notna().mean() * 100) if len(data) else 0.0})
    return pd.DataFrame(rows)


def sensor_outages(data: pd.DataFrame, minimum_minutes: int = 1) -> pd.DataFrame:
    rows = []
    for column in [c for c in data if c.endswith(("_kw", "_mw")) or c == "temperature_c"]:
        flag_column = _provenance_flag(column)
        missing = data[flag_column].astype(bool) if flag_column in data else data[column].isna()
        groups = missing.ne(missing.shift()).cumsum()
        for _, part in data.loc[missing, ["timestamp"]].groupby(groups[missing]):
            duration = len(part)
            if duration >= minimum_minutes:
                rows.append({"channel": column, "start": part.timestamp.iloc[0], "end": part.timestamp.iloc[-1],
                             "duration_minutes": duration})
    if not rows:
        return pd.DataFrame(columns=["channel", "start", "end", "duration_minutes"])
    return pd.DataFrame(rows).sort_values("duration_minutes", ascending=False, ignore_index=True)


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in data if c.endswith(("_kw", "_mw")) or c == "temperature_c"]
    if not columns:
        return pd.DataFrame()
    return data[columns].describe().T.reset_index(names="channel")


def channel_catalog(data: pd.DataFrame) -> pd.DataFrame:
    scope = lambda c: "Household" if c.endswith("_kw") else ("Regional" if c.endswith("_mw") else "Context/label")
    unit = lambda c: "kW" if c.endswith("_kw") else ("MW" if c.endswith("_mw") else ("°C" if c == "temperature_c" else "—"))
    return pd.DataFrame({"channel": data.columns, "scope": [scope(c) for c in data],
                         "unit": [unit(c) for c in data], "dtype": [str(data[c].dtype) for c in data]})
