"""Application service for historical and boundary day-ahead forecasts."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from backend.data.aggregation import operational_household
from backend.data.v3_loader import load_household
from backend.intelligence.forecast_features import build_window
from backend.intelligence.forecast_metrics import forecast_kpis, forecast_metrics
from backend.intelligence.forecasting import load_forecaster
from backend.intelligence.model_registry import DEFAULT_FORECAST_ARTIFACT, ModelRegistry


class Predictor(Protocol):
    metadata: dict
    def predict(self, context_kw: np.ndarray) -> np.ndarray: ...


def forecast_next_day(client_id: str, forecast_date: object, *, predictor: Predictor | None = None,
                      artifact_path: Path | str = DEFAULT_FORECAST_ARTIFACT) -> dict:
    day = pd.Timestamp(forecast_date).normalize()
    raw = load_household(client_id, day - pd.DateOffset(days=7), day + pd.DateOffset(days=1))
    operational = operational_household(raw)
    window = build_window(operational, client_id, day)
    model = predictor or load_forecaster(str(artifact_path))
    prediction_kw = np.asarray(model.predict(window.context_kw), dtype=np.float32)
    if prediction_kw.shape != (48,):
        raise ValueError(f"Model returned {len(prediction_kw)} predictions; exactly 48 are required.")
    frame = pd.DataFrame({
        "timestamp": window.target_timestamps,
        "predicted_power_kw": prediction_kw,
        "predicted_energy_kwh": prediction_kw * 0.5,
    })
    if window.target_kw is not None:
        frame["actual_power_kw"] = window.target_kw
        frame["actual_energy_kwh"] = window.target_kw * 0.5
    metadata = model.metadata
    test_start = pd.Timestamp(metadata["test_start"]).normalize()
    comparison = "held_out" if window.target_kw is not None and day >= test_start else (
        "in_sample" if window.target_kw is not None else "unavailable")
    metrics = forecast_metrics(window.target_kw, prediction_kw) if comparison == "held_out" else None
    return {
        "client_id": client_id,
        "forecast_date": day,
        "forecast_issue_time": day,
        "forecast_timestamps": window.target_timestamps,
        "target_unit": "kW (power); kWh per 30-minute interval derived as kW x 0.5 h",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "historical_input_window": {
            "start": window.context_timestamps[0], "end": window.context_timestamps[-1],
            "slots": len(window.context_kw),
        },
        "actual_comparison": comparison,
        "uncertainty": None,
        "series": frame,
        "kpis": forecast_kpis(prediction_kw),
        "metrics": metrics,
    }


def model_status(artifact_path: Path | str = DEFAULT_FORECAST_ARTIFACT) -> tuple[bool, str]:
    registry = ModelRegistry(Path(artifact_path))
    try:
        metadata = registry.metadata()
        return True, f"{metadata['model_name']} {metadata['model_version']}"
    except Exception as exc:
        return False, str(exc)
