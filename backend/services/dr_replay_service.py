"""Operator-facing access to saved historical replay artifacts."""
from __future__ import annotations

import json

import pandas as pd

from backend.intelligence.community_forecasting import COMMUNITY_ARTIFACT, artifact_status
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
    return pd.date_range("2025-07-01", "2025-08-31", freq="1D")
