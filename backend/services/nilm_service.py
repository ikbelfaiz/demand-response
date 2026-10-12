"""Application service for historical, aggregate-only TCN NILM inference."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

from backend.config import (nilm_checkpoint_path, nilm_data_signature, nilm_device,
                            nilm_max_request_minutes)
from backend.data import load_household_info
from backend.intelligence.nilm import (NILMArtifactError, artifact_identity,
                                       load_nilm_runtime)
from nilm_research.data import NILMRepository, endpoint_quality, prepare_partition
from nilm_research.evaluate import predict_partition

APPLIANCES = ("ac", "water_heater", "washing_machine")
DISPLAY_NAMES = {"ac": "Primary AC", "water_heater": "Water heater",
                 "washing_machine": "Washing machine"}
LOCAL_TIMEZONE = "Africa/Tunis"
DATA_START = pd.Timestamp("2025-01-01 00:00:00")
DATA_END = pd.Timestamp("2026-01-01 00:00:00")


@dataclass
class NILMResult:
    request: dict
    model: dict
    minute: pd.DataFrame
    hourly: pd.DataFrame
    summary: dict
    latency_seconds: float

    def json_safe(self) -> dict:
        def clean(value):
            if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)): return [clean(v) for v in value]
            if isinstance(value, pd.Timestamp):
                ts = value.tz_localize(LOCAL_TIMEZONE) if value.tzinfo is None else value.tz_convert(LOCAL_TIMEZONE)
                return ts.isoformat()
            if isinstance(value, (np.integer,)): return int(value)
            if isinstance(value, (np.floating, float)):
                return None if not math.isfinite(float(value)) else float(value)
            if isinstance(value, np.bool_): return bool(value)
            if pd.isna(value): return None
            return value
        return {
            "request": clean(self.request), "model": clean(self.model), "summary": clean(self.summary),
            "latency_seconds": self.latency_seconds,
            "minute_results": [{k: clean(v) for k, v in row.items()} for row in self.minute.to_dict("records")],
            "hourly_results": [{k: clean(v) for k, v in row.items()} for row in self.hourly.to_dict("records")],
        }


def _local_naive(value: object, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(LOCAL_TIMEZONE).tz_localize(None)
    if ts.second or ts.microsecond or ts.nanosecond:
        raise ValueError(f"{label} must align to a whole minute")
    return ts


def _validate_request(client_id: str, start: object, end: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    known = set(load_household_info().client_id.astype(str))
    if client_id not in known:
        raise ValueError(f"Unknown household {client_id!r}")
    start_ts, end_ts = _local_naive(start, "start"), _local_naive(end, "end")
    if start_ts >= end_ts: raise ValueError("start must be before end; intervals are [start, end)")
    minutes = int((end_ts - start_ts).total_seconds() // 60)
    if minutes > nilm_max_request_minutes():
        raise ValueError(f"Requested {minutes} minutes; limit is {nilm_max_request_minutes()}")
    if start_ts < DATA_START or end_ts > DATA_END:
        raise ValueError("NILM requests must be within the historical 2025 dataset")
    return start_ts, end_ts


def _hourly(minute: pd.DataFrame) -> pd.DataFrame:
    work = minute.copy()
    work["hour"] = work.timestamp.dt.floor("h")
    rows = []
    for hour, group in work.groupby("hour", sort=True):
        valid = group.quality_status.eq("ok")
        row = {"hour": hour, "requested_minutes": int(len(group)),
               "valid_minutes": int(valid.sum()),
               "coverage_fraction": float(valid.mean()) if len(group) else 0.0,
               "full_hour_coverage_fraction": float(valid.sum() / 60.0)}
        for appliance in APPLIANCES:
            values = group.loc[valid, f"predicted_{appliance}_power_w"]
            row[f"{appliance}_energy_kwh"] = float(values.sum() / 60000.0) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(minute: pd.DataFrame) -> dict:
    valid = minute.quality_status.eq("ok")
    expected = len(minute)
    appliances = {}
    for appliance in APPLIANCES:
        power_col, probability_col = (f"predicted_{appliance}_power_w",
                                      f"{appliance}_activity_probability")
        usable = minute.loc[valid & minute[power_col].notna()]
        latest = usable.iloc[-1] if len(usable) else None
        appliances[appliance] = {
            "display_name": DISPLAY_NAMES[appliance],
            "observed_estimated_energy_kwh": float(usable[power_col].sum() / 60000.0) if len(usable) else None,
            "coverage_fraction": float(len(usable) / expected) if expected else 0.0,
            "latest_valid_power_w": float(latest[power_col]) if latest is not None else None,
            "latest_valid_timestamp": latest.timestamp if latest is not None else None,
            "latest_activity_probability": float(latest[probability_col]) if latest is not None else None,
        }
    return {
        "expected_minutes": expected, "available_readings": int(minute.aggregate_power_w.notna().sum()),
        "valid_predicted_minutes": int(valid.sum()),
        "coverage_fraction": float(valid.mean()) if expected else 0.0,
        "excluded_minutes_by_reason": {str(k): int(v) for k, v in minute.loc[~valid, "quality_reason"].value_counts().items()},
        "energy_rule": "sum(valid one-minute average power W) / 60000; no extrapolation",
        "appliances": appliances,
    }


def _cfg_copy(runtime) -> dict:
    return {key: deepcopy(value) for key, value in runtime.config.items()}


@lru_cache(maxsize=16)
def _predict_cached(checkpoint_identity, data_identity, requested_device: str,
                    client_id: str, start_text: str, end_text: str) -> NILMResult:
    del data_identity  # Included in the cache key to invalidate replaced source files.
    started = time.perf_counter()
    runtime = load_nilm_runtime(checkpoint_identity[0], requested_device)
    start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
    context = int(runtime.metadata["context_length"])
    history_start = start - timedelta(minutes=context - 1)
    cfg = _cfg_copy(runtime)
    cfg["splits"]["application_inference"] = [str(history_start), str(end)]
    part = prepare_partition(NILMRepository(cfg), cfg, dict(runtime.transforms), client_id,
                             "application_inference")
    power, probability, endpoint = predict_partition(
        runtime.model, part, context, int(cfg["evaluation"]["sequence_chunk_size"]), runtime.device)
    _, reasons = endpoint_quality(part, context)
    selected = (part.timestamps >= np.datetime64(start)) & (part.timestamps < np.datetime64(end))
    idx = np.flatnonzero(selected)
    valid = endpoint[idx]
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(part.timestamps[idx]), "client_id": client_id,
        "aggregate_power_w": part.aggregate_w[idx],
        "quality_status": np.where(valid, "ok", "unavailable"),
        "quality_reason": reasons[idx], "is_dr_event": part.dr_event[idx],
    })
    for j, appliance in enumerate(APPLIANCES):
        frame[f"predicted_{appliance}_power_w"] = power[idx, j]
        frame[f"{appliance}_activity_probability"] = probability[idx, j]
        reference = np.where(part.target_valid[idx, j], part.targets_w[idx, j], np.nan)
        frame[f"reference_{appliance}_power_w"] = reference
        frame[f"reference_{appliance}_activity"] = np.where(
            part.target_valid[idx, j], part.activity[idx, j], np.nan)
    expected_timestamps = pd.date_range(start, end, freq="min", inclusive="left")
    frame = (frame.set_index("timestamp").reindex(expected_timestamps)
             .rename_axis("timestamp").reset_index())
    frame["client_id"] = frame.client_id.fillna(client_id)
    absent = frame.quality_status.isna()
    frame.loc[absent, "quality_status"] = "unavailable"
    frame.loc[absent, "quality_reason"] = "missing_or_imputed_input"
    predicted_sum = frame[[f"predicted_{a}_power_w" for a in APPLIANCES]].sum(axis=1, min_count=3)
    frame["diagnostic_residual_w"] = frame.aggregate_power_w - predicted_sum
    hourly = _hourly(frame)
    request = {"household_id": client_id, "start": start, "end": end,
               "interval_semantics": "[start, end)", "timezone": LOCAL_TIMEZONE,
               "display_resolution": "1min"}
    return NILMResult(request, dict(runtime.metadata), frame, hourly, _summary(frame),
                      time.perf_counter() - started)


def predict_appliances(client_id: str, start: object, end: object,
                       *, checkpoint: Path | str | None = None,
                       device: str | None = None) -> NILMResult:
    """Predict at one-minute resolution; references are optional and never inputs."""
    start_ts, end_ts = _validate_request(client_id, start, end)
    path = Path(checkpoint) if checkpoint is not None else nilm_checkpoint_path()
    identity = artifact_identity(path) if path.is_file() else (str(path), 0, 0)
    result = _predict_cached(identity, nilm_data_signature(), device or nilm_device(),
                             client_id, str(start_ts), str(end_ts))
    return NILMResult(deepcopy(result.request), deepcopy(result.model), result.minute.copy(deep=True),
                      result.hourly.copy(deep=True), deepcopy(result.summary), result.latency_seconds)


def model_status(*, checkpoint: Path | str | None = None, device: str | None = None) -> tuple[bool, dict | str]:
    path = Path(checkpoint) if checkpoint is not None else nilm_checkpoint_path()
    try:
        runtime = load_nilm_runtime(path, device or nilm_device())
        metadata = dict(runtime.metadata)
        metrics_path = path.with_name("metrics_test.csv")
        if metrics_path.is_file():
            metrics = pd.read_csv(metrics_path)
            rows = metrics[(metrics.household == "ALL") & (metrics.stratum == "overall") & (metrics.stratum_value == "all")]
            metadata["verified_pooled_test_metrics"] = {
                row.appliance: {"power_mae_w": float(row.power_mae_w), "activity_f1": float(row.activity_f1)}
                for row in rows.itertuples()
            }
        return True, metadata
    except Exception as exc:
        return False, str(exc)


def export_csv(result: NILMResult, kind: str = "minute") -> bytes:
    if kind not in {"minute", "hourly"}: raise ValueError("kind must be minute or hourly")
    frame = result.minute if kind == "minute" else result.hourly
    return frame.to_csv(index=False, na_rep="").encode("utf-8")
