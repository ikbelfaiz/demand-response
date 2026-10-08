"""Deterministic forecast-margin risk detection; no fabricated probabilities."""
from __future__ import annotations

import pandas as pd


def _groups(mask: pd.Series) -> list[list[int]]:
    groups, current = [], []
    for position, active in enumerate(mask.fillna(False).astype(bool)):
        if active:
            current.append(position)
        elif current:
            groups.append(current); current = []
    if current: groups.append(current)
    return groups


def detect_risk_windows(forecast: pd.DataFrame, *, safety_margin_kw: float = 0.0) -> pd.DataFrame:
    required = {"timestamp","predicted_demand_kw","predicted_supply_kw","predicted_margin_kw"}
    if not required.issubset(forecast):
        raise ValueError(f"Missing forecast columns: {sorted(required-set(forecast))}")
    deficit = (-forecast.predicted_margin_kw - safety_margin_kw).clip(lower=0)
    rows = []
    for risk_id, positions in enumerate(_groups(deficit.gt(0)), 1):
        block = forecast.iloc[positions].copy()
        block_deficit = deficit.iloc[positions]
        peak_position = int(block_deficit.to_numpy().argmax())
        rows.append({
            "risk_id": risk_id,
            "start_ts": block.timestamp.iloc[0],
            "end_ts": block.timestamp.iloc[-1] + pd.offsets.Minute(30),
            "duration_minutes": len(block) * 30,
            "max_shortfall_kw": float(block_deficit.max()),
            "shortage_energy_kwh": float(block_deficit.sum() * 0.5),
            "peak_shortfall_ts": block.timestamp.iloc[peak_position],
            "mean_demand_kw": float(block.predicted_demand_kw.mean()),
            "mean_supply_kw": float(block.predicted_supply_kw.mean()),
            "slot_count": len(block),
        })
    return pd.DataFrame(rows, columns=["risk_id","start_ts","end_ts","duration_minutes",
        "max_shortfall_kw","shortage_energy_kwh","peak_shortfall_ts","mean_demand_kw",
        "mean_supply_kw","slot_count"])


def boolean_windows(timestamps, labels) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp":pd.to_datetime(timestamps),"label":labels})
    rows = []
    for event_id, positions in enumerate(_groups(frame.label.astype(bool)), 1):
        block = frame.iloc[positions]
        rows.append({"event_id":event_id,"start_ts":block.timestamp.iloc[0],
                     "end_ts":block.timestamp.iloc[-1]+pd.offsets.Minute(30),
                     "intervals":len(block)})
    return pd.DataFrame(rows, columns=["event_id","start_ts","end_ts","intervals"])
