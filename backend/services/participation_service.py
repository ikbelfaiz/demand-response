"""Participation summaries isolated from research counterfactual truth."""
import pandas as pd


def for_household(participation: pd.DataFrame, events: pd.DataFrame, client_id: str) -> pd.DataFrame:
    return participation.loc[participation.client_id == client_id].merge(events, on="event_id", how="left").sort_values("start_ts")


def response_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.response.value_counts()
    return {name: int(counts.get(name, 0)) for name in ("accept", "decline", "no_response")}
