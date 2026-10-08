"""Historical DR event exploration; no model-generated events."""
import pandas as pd


def event_table(events: pd.DataFrame, participation: pd.DataFrame) -> pd.DataFrame:
    counts = participation.pivot_table(index="event_id", columns="response", values="client_id", aggfunc="count", fill_value=0)
    for column in ("accept", "decline", "no_response"):
        if column not in counts: counts[column] = 0
    counts = counts.reset_index()
    out = events.merge(counts, on="event_id", how="left")
    out["invited"] = out[["accept", "decline", "no_response"]].sum(axis=1)
    out["participation_pct"] = out["accept"] / out["invited"] * 100
    out["duration_minutes"] = (out.end_ts - out.start_ts).dt.total_seconds() / 60
    return out


def event_window(grid: pd.DataFrame, event: pd.Series, padding_hours: int = 3) -> pd.DataFrame:
    start, end = event.start_ts - pd.DateOffset(hours=padding_hours), event.end_ts + pd.DateOffset(hours=padding_hours)
    return grid.loc[(grid.timestamp >= start) & (grid.timestamp <= end)].copy()
