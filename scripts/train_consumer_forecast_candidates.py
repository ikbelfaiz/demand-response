"""Train isolated consumer Energy-TTM candidates without touching active/operator artifacts."""
from __future__ import annotations

import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

from backend.intelligence.consumer_forecast_data import build_consumer_window, load_causal_household
from backend.intelligence.forecast_features import normalize_context
from scripts.train_forecasting import CACHE_SNAPSHOT, CHECKPOINT

OUT = ROOT / "models" / "household_forecasting" / "candidates"
TRAIN_DATES = pd.date_range("2025-01-08", "2025-09-30", freq="D")
VALIDATION_DATES = pd.date_range("2025-10-01", "2025-10-31", freq="D")
SPECS = (
    {"name": "energy_ttm_causal_mse", "peak_weight": 1.0},
    {"name": "energy_ttm_causal_peak", "peak_weight": 3.0},
)


def prepare(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    contexts, targets, records = [], [], []
    for number in range(1, 51):
        client = f"C{number:03d}"; frame = load_causal_household(client)
        for day in dates:
            try:
                window = build_consumer_window(frame, client, day, require_target=True)
            except ValueError:
                continue
            scaled, mean, std = normalize_context(window.context_kw)
            contexts.append(scaled[:, None])
            targets.append(((window.target_kw - mean) / (std + 1e-8))[:, None])
            records.append({"client_id": client, "forecast_date": str(day.date())})
        print(f"Prepared {client} ({number}/50)", flush=True)
    return np.asarray(contexts, np.float32), np.asarray(targets, np.float32), records


def batch_loss(model, context, target, peak_weight: float) -> torch.Tensor:
    prediction = model(past_values=context).prediction_outputs[:, :48, :]
    squared = (prediction - target) ** 2
    if peak_weight > 1:
        # Weight the six highest observed half-hours per day. Ranking uses only
        # the training target and never becomes an inference feature.
        peak_index = target[..., 0].topk(k=6, dim=1).indices
        weights = torch.ones_like(target[..., 0])
        weights.scatter_(1, peak_index, peak_weight)
        squared = squared * weights[..., None]
    return squared.mean()


def epoch(model, loader, device, peak_weight: float, optimizer=None) -> float:
    model.train(optimizer is not None); losses = []
    for context, target in loader:
        context, target = context.to(device), target.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, context, target, peak_weight if optimizer is not None else 1.0)
        if optimizer is not None:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train(spec: dict, train_x, train_y, val_x, val_y, train_records, val_records) -> None:
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    source = str(CACHE_SNAPSHOT) if CACHE_SNAPSHOT.is_dir() else CHECKPOINT
    model = TinyTimeMixerForPrediction.from_pretrained(
        source, local_files_only=CACHE_SNAPSHOT.is_dir(), ignore_mismatched_sizes=True,
        context_length=336, prediction_length=48, patch_length=48, patch_stride=48,
        num_input_channels=1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                              batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                            batch_size=64, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for number in range(1, 11):
        train_loss = epoch(model, train_loader, device, spec["peak_weight"], optimizer)
        with torch.inference_mode():
            validation_loss = epoch(model, val_loader, device, 1.0)
        history.append({"epoch": number, "train_objective": train_loss,
                        "validation_mse": validation_loss})
        print(f"{spec['name']} epoch {number}: train={train_loss:.6f} val={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-4:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 3:
                break
    model.load_state_dict(best_state)
    artifact = OUT / spec["name"]; model_dir = artifact / "model"; model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    metadata = {
        "model_name": "Consumer Energy-TTM leakage-safe candidate",
        "model_version": spec["name"], "architecture": "TinyTimeMixerForPrediction",
        "base_checkpoint": CHECKPOINT, "context_length": 336, "prediction_length": 48,
        "frequency": "30min", "target": "aggregate_power_kw", "model_directory": "model",
        "test_start": "2025-11-01", "households": [f"C{i:03d}" for i in range(1, 51)],
        "source_files": ["data/original_v3_backup/households_1min_C001_C025.parquet",
                         "data/original_v3_backup/households_1min_C026_C050.parquet"],
        "preprocessing": "30-minute mean with >=80% minute coverage; contexts forward-filled from past only; incomplete targets excluded",
        "normalization": "per-origin mean/std from the 336 causal historical values only",
        "patch_length": 48, "patch_stride": 48, "direct_output_slots": 48,
        "loss": "MSE" if spec["peak_weight"] == 1 else "MSE with 3x weight on six highest target slots",
        "learning_rate": 3e-4, "max_epochs": 10, "early_stopping_patience": 3,
        "selected_epoch": int(pd.DataFrame(history).validation_mse.idxmin() + 1),
        "best_validation_mse_normalized": best_loss,
        "split": {"train": "2025-01-08/2025-09-30", "validation": "2025-10-01/2025-10-31",
                  "test": "2025-11-01/2025-12-31"},
        "training_windows": len(train_records), "validation_windows": len(val_records),
        "excluded_incomplete_train_targets": 50 * len(TRAIN_DATES) - len(train_records),
        "excluded_incomplete_validation_targets": 50 * len(VALIDATION_DATES) - len(val_records),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(), "device": device, "random_seed": 42,
        "status": "candidate_not_active",
    }
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(artifact / "training_history.csv", index=False)
    pd.DataFrame(train_records).to_csv(artifact / "train_windows.csv", index=False)
    pd.DataFrame(val_records).to_csv(artifact / "validation_windows.csv", index=False)


def main() -> None:
    train_x, train_y, train_records = prepare(TRAIN_DATES)
    val_x, val_y, val_records = prepare(VALIDATION_DATES)
    print(f"Eligible windows: train={len(train_records)}, validation={len(val_records)}", flush=True)
    for spec in SPECS:
        train(spec, train_x, train_y, val_x, val_y, train_records, val_records)


if __name__ == "__main__":
    main()
