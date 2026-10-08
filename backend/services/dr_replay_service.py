"""Operator-facing access to saved historical replay artifacts."""
from __future__ import annotations

import json

import pandas as pd

from backend.intelligence.community_forecasting import (
    COMMUNITY_ARTIFACT, CommunityForecastBundle, artifact_status,
    community_forecast_classification, eligible_community_forecast_dates,
)
from backend.intelligence.decision_engine import DecisionPolicy
from backend.intelligence.historical_replay import replay_day


def load_policy() -> DecisionPolicy:
    path = COMMUNITY_ARTIFACT / "decision_policy.json"
    if not path.is_file(): return DecisionPolicy()
    return DecisionPolicy(**json.loads(path.read_text(encoding="utf-8")))


def run_replay(target_date: object) -> dict:
    return replay_day(target_date, policy=load_policy())


def replay_model_status() -> tuple[bool, str]:
    return artifact_status()


def load_backtest_summary() -> dict | None:
    path = COMMUNITY_ARTIFACT / "backtest_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def held_out_dates() -> pd.DatetimeIndex:
    metadata = CommunityForecastBundle().metadata
    test = metadata.get("split", {}).get("test")
    if not test:
        return pd.DatetimeIndex([])
    start, end = test.split("/", maxsplit=1)
    return pd.date_range(start, end, freq="1D")


def eligible_replay_dates() -> pd.DatetimeIndex:
    return eligible_community_forecast_dates()


def replay_classification(target_date: object) -> tuple[str, str]:
    return community_forecast_classification(CommunityForecastBundle().metadata, target_date)
