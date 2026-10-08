"""Research-only DR recommendation policy, separate from forecasting and activation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class DecisionPolicy:
    min_deficit_kw: float = 1.0
    min_duration_minutes: int = 60
    min_shortage_energy_kwh: float = 1.0
    safety_margin_kw: float = 0.0
    max_event_duration_minutes: int = 120
    min_gap_minutes: int = 60
    max_events_per_day: int = 2
    version: str = "research-policy-v1"


def _bounded_window(forecast: pd.DataFrame, risk: pd.Series, policy: DecisionPolicy) -> pd.DataFrame:
    block = forecast.loc[(forecast.timestamp >= risk.start_ts) & (forecast.timestamp < risk.end_ts)].copy()
    max_slots = max(1, policy.max_event_duration_minutes // 30)
    if len(block) <= max_slots:
        return block
    peak = int((-block.predicted_margin_kw).to_numpy().argmax())
    start = min(max(0, peak - max_slots // 2), len(block) - max_slots)
    return block.iloc[start:start + max_slots]


def recommend_events(forecast: pd.DataFrame, risks: pd.DataFrame, *, issue_time: object,
                     model_version: str, policy: DecisionPolicy = DecisionPolicy()) -> pd.DataFrame:
    rows = []
    for _, risk in risks.sort_values(["max_shortfall_kw","shortage_energy_kwh"], ascending=False).iterrows():
        if risk.max_shortfall_kw < policy.min_deficit_kw or risk.duration_minutes < policy.min_duration_minutes \
                or risk.shortage_energy_kwh < policy.min_shortage_energy_kwh:
            continue
        block = _bounded_window(forecast, risk, policy)
        start, end = block.timestamp.iloc[0], block.timestamp.iloc[-1] + pd.offsets.Minute(30)
        if any(not (end + pd.offsets.Minute(policy.min_gap_minutes) <= row["start_ts"] or
                       start >= row["end_ts"] + pd.offsets.Minute(policy.min_gap_minutes)) for row in rows):
            continue
        deficit = (-block.predicted_margin_kw).clip(lower=0)
        max_deficit = float(deficit.max())
        mean_demand = float(block.predicted_demand_kw.mean())
        rows.append({
            "start_ts":start,"end_ts":end,"forecast_created_at":pd.Timestamp(issue_time),
            "predicted_demand_kw":mean_demand,"predicted_supply_kw":float(block.predicted_supply_kw.mean()),
            "max_deficit_kw":max_deficit,"shortage_energy_kwh":float(deficit.sum()*0.5),
            "duration_minutes":int(len(block)*30),"recommended_reduction_kw":max_deficit,
            "recommended_reduction_pct":min(100.0, max_deficit/mean_demand*100) if mean_demand>0 else 0.0,
            "decision_status":"Simulated recommendation — historical replay only",
            "reason":f"Forecast deficit met {policy.min_deficit_kw:g} kW, {policy.min_duration_minutes} min, and {policy.min_shortage_energy_kwh:g} kWh research thresholds.",
            "forecast_model_version":model_version,"decision_policy_version":policy.version,
        })
        if len(rows) >= policy.max_events_per_day:
            break
    if not rows:
        return pd.DataFrame(columns=["start_ts","end_ts","forecast_created_at","predicted_demand_kw",
            "predicted_supply_kw","max_deficit_kw","shortage_energy_kwh","duration_minutes",
            "recommended_reduction_kw","recommended_reduction_pct","decision_status","reason",
            "forecast_model_version","decision_policy_version"])
    return pd.DataFrame(rows).sort_values("start_ts", ignore_index=True)


def policy_metadata(policy: DecisionPolicy) -> dict:
    return asdict(policy)


def propose_event(*args, **kwargs):
    """Backward-compatible alias for research recommendations; never activates an event."""
    return recommend_events(*args, **kwargs)
