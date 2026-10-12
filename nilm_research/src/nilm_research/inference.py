from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .data import NILMRepository, endpoint_quality, prepare_partition
from .evaluate import predict_partition


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def hourly_energy_rows(timestamps: np.ndarray, power_w: np.ndarray, valid: np.ndarray) -> list[dict]:
    rows = []
    if not len(timestamps):
        return rows
    hours = timestamps.astype("datetime64[h]")
    for hour in np.unique(hours):
        loc = hours == hour
        usable = loc & valid
        count = int(usable.sum())
        energy = np.nansum(power_w[usable], axis=0) / 60000.0 if count else [None] * 3
        requested = int(loc.sum())
        rows.append({"hour": str(hour).replace("T", " "), "valid_minutes": count,
                     "requested_minutes": requested,
                     "coverage_fraction": count / requested if requested else 0.0,
                     "full_hour_coverage_fraction": count / 60.0,
                     "ac_energy_kwh": energy[0], "water_heater_energy_kwh": energy[1],
                     "washing_machine_energy_kwh": energy[2]})
    return rows


def run_inference(checkpoint: str | Path, household: str, start: str, end: str,
                  output_csv: str | Path, device_name: str = "auto") -> dict:
    model, payload = load_checkpoint(checkpoint, "cpu")
    cfg, transforms = deepcopy(payload["config"]), payload["transforms"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          ("cpu" if device_name == "auto" else device_name))
    model.to(device)
    requested_start, requested_end = _parse(start), _parse(end)
    context = int(cfg["data"]["context_length"])
    history_start = requested_start - timedelta(minutes=context - 1)
    cfg["splits"]["inference"] = [history_start.isoformat(sep=" "), requested_end.isoformat(sep=" ")]
    repo = NILMRepository(cfg)
    part = prepare_partition(repo, cfg, transforms, household, "inference")
    power, probability, endpoint = predict_partition(
        model, part, context, int(cfg["evaluation"]["sequence_chunk_size"]), device
    )
    selected = (part.timestamps >= np.datetime64(requested_start)) & (part.timestamps < np.datetime64(requested_end))
    indices = np.flatnonzero(selected)
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    _, reasons = endpoint_quality(part, context)
    fields = ["timestamp", "client_id", "aggregate_power_w",
              "ac_power_w_predicted", "water_heater_power_w_predicted", "washing_machine_power_w_predicted",
              "ac_power_w_true", "water_heater_power_w_true", "washing_machine_power_w_true",
              "ac_activity_probability", "water_heater_activity_probability", "washing_machine_activity_probability",
              "ac_activity_reference", "water_heater_activity_reference", "washing_machine_activity_reference",
              "diagnostic_residual_w", "quality_status"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in indices:
            if not endpoint[i]:
                status = str(reasons[i])
                values = [None] * 13
            else:
                status = "ok"
                residual = float(part.aggregate_w[i] - np.nansum(power[i]))
                true = [float(part.targets_w[i, j]) if part.target_valid[i, j] else None for j in range(3)]
                reference = [int(part.activity[i, j]) if part.target_valid[i, j] else None for j in range(3)]
                values = [*map(float, power[i]), *true, *map(float, probability[i]), *reference, residual]
            writer.writerow(dict(zip(fields, [str(part.timestamps[i]).replace("T", " "), household,
                                               float(part.aggregate_w[i]), *values, status])))

    hourly_path = target.with_name(target.stem + "_hourly.csv")
    with hourly_path.open("w", newline="", encoding="utf-8") as handle:
        fields_h = ["hour", "client_id", "valid_minutes", "requested_minutes", "coverage_fraction", "full_hour_coverage_fraction",
                    "ac_energy_kwh", "water_heater_energy_kwh", "washing_machine_energy_kwh"]
        writer = csv.DictWriter(handle, fieldnames=fields_h); writer.writeheader()
        for row in hourly_energy_rows(part.timestamps[indices], power[indices], endpoint[indices]):
            writer.writerow({"client_id": household, **row})
    status = {
        "household": household, "requested_start": start, "requested_end": end,
        "rows": int(len(indices)), "valid_predictions": int(endpoint[indices].sum()),
        "insufficient_or_contaminated": int((~endpoint[indices]).sum()),
        "aggregate_imputed_points_loaded": int(part.aggregate_imputed.sum()),
        "aggregate_spikes_loaded": int(part.aggregate_spike.sum()),
        "frozen_points_loaded": int(part.frozen.sum()),
        "prediction_csv": str(target.resolve()), "hourly_csv": str(hourly_path.resolve()),
        "hourly_energy_rule": "sum(valid minute-average W)/60000; no extrapolation",
    }
    return status
