"""Event decision: takes the team's recommended DR events and adds the surcharge level.

The window comes from backend.intelligence.decision_engine.recommend_events (saved research policy).
The orchestrator keeps the recommendation with the largest expected shortage, enforces the weekly
limit, and sets the surcharge level from the share of supply that is missing in the window.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.orchestration import settings


@dataclass
class Decision:
    start: pd.Timestamp
    end: pd.Timestamp
    level: int
    max_deficit_kw: float
    shortage_energy_kwh: float
    deficit_share: float
    recommended_reduction_pct: float


def choose_level(deficit_share: float, outage: bool = False) -> int:
    if outage or deficit_share >= settings.LEVEL3_DEFICIT_SHARE:
        return 3
    if deficit_share < settings.LEVEL1_DEFICIT_SHARE:
        return 1
    return 2


def decide(forecast: pd.DataFrame, recommendations: pd.DataFrame, events_this_week: int,
           outage: bool = False) -> Decision | None:
    if recommendations is None or recommendations.empty or events_this_week >= settings.MAX_EVENTS_PER_WEEK:
        return None
    rec = recommendations.sort_values("shortage_energy_kwh", ascending=False).iloc[0]
    start, end = pd.Timestamp(rec.start_ts), pd.Timestamp(rec.end_ts)
    window = forecast[(forecast.timestamp >= start) & (forecast.timestamp < end)]
    supply_kwh = float((window.predicted_supply_kw * settings.SLOT_HOURS).sum())
    share = float(rec.shortage_energy_kwh) / supply_kwh if supply_kwh > 0 else 0.0
    return Decision(start=start, end=end, level=choose_level(share, outage),
                    max_deficit_kw=round(float(rec.max_deficit_kw), 3),
                    shortage_energy_kwh=round(float(rec.shortage_energy_kwh), 3),
                    deficit_share=round(share, 4),
                    recommended_reduction_pct=round(float(rec.recommended_reduction_pct), 2))
