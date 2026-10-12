"""Adapter to the team's community DR forecast (Ridge profile model, issued at D-1 14:00).

Uses, without changing them:
  backend.intelligence.community_forecasting  → CommunityForecastBundle.predict_day()
  backend.intelligence.peak_detection         → detect_risk_windows()
  backend.intelligence.decision_engine        → recommend_events() with the saved research policy
The model is deterministic (no probability): a risk window is a run of negative forecast margins.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from backend.intelligence.community_forecasting import (
    COMMUNITY_ARTIFACT, eligible_community_forecast_dates, load_causal_operational_grid, load_community_bundle,
)
from backend.intelligence.decision_engine import recommend_events
from backend.intelligence.peak_detection import detect_risk_windows
from backend.services.dr_replay_service import load_policy


@dataclass
class DayAhead:
    target_day: pd.Timestamp
    issue_time: pd.Timestamp
    forecast: pd.DataFrame          # timestamp, predicted_demand_kw, predicted_supply_kw, predicted_margin_kw, …
    risks: pd.DataFrame             # detect_risk_windows output
    recommendations: pd.DataFrame   # recommend_events output (start_ts, end_ts, max_deficit_kw, …)
    model_version: str
    policy_version: str


class CommunityDRForecaster:
    def __init__(self):
        self.bundle = load_community_bundle(str(COMMUNITY_ARTIFACT))
        self.policy = load_policy()
        self.eligible = set(eligible_community_forecast_dates())
        self.causal = load_causal_operational_grid()

    def day_ahead(self, decision_day: date) -> DayAhead | None:
        """Run on D-1 at 14:30 for the next day. None if the model cannot forecast that day."""
        target = pd.Timestamp(decision_day).normalize() + pd.Timedelta(days=1)
        if target not in self.eligible:
            return None
        issue = target - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        forecast = self.bundle.predict_day(self.causal, target)
        risks = detect_risk_windows(forecast, safety_margin_kw=self.policy.safety_margin_kw)
        version = self.bundle.metadata["model_version"]
        recs = recommend_events(forecast, risks, issue_time=issue, model_version=version, policy=self.policy)
        return DayAhead(target, issue, forecast, risks, recs, version, self.policy.version)
