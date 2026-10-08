"""Fine-tune the notebook's Energy-TTM model for 336-to-48 household forecasting."""
from __future__ import annotations

import argparse
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

from backend.data import load_household, operational_household
from backend.intelligence.forecast_features import build_window, normalize_context
from backend.intelligence.model_registry import DEFAULT_FORECAST_ARTIFACT

CHECKPOINT = "EnergyFM/energy-ttm"
CACHE_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--EnergyFM--energy-ttm" / "snapshots" / "63acc54104d46c284e54ef92812b3d2aa533cdea"
TRAIN_END = pd.Timestamp("2025-09-30")
VALIDATION_END = pd.Timestamp("2025-10-31")


def household_ids(count: int) -> list[str]:
    if not 1 <= count <= 50:
        raise ValueError("households must be between 1 and 50")
    return [f"C{i:03d}" for i in range(1, count + 1)]


def prepare_split(clients: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    contexts, targets, records = [], [], []
    for number, client_id in enumerate(clients, 1):
        raw = load_household(client_id, start - pd.DateOffset(days=7), end + pd.DateOffset(days=1),
                             columns=["timestamp", "client_id", "aggregate_power_w"])
        operational = operational_household(raw)
        for day in pd.date_range(start, end, freq="1D"):
            window = build_window(operational, client_id, day, require_target=True)
            scaled, mean, std = normalize_context(window.context_kw)
            contexts.append(scaled[:, None])
            targets.append(((window.target_kw - mean) / (std + 1e-8))[:, None])
            records.append({"client_id": client_id, "forecast_date": str(day.date())})
        print(f"Prepared {client_id} ({number}/{len(clients)})", flush=True)
    return np.asarray(contexts, np.float32), np.asarray(targets, np.float32), records


def run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    for context, target in loader:
        context, target = context.to(device), target.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        output = model(past_values=context, future_values=target)
        loss = output.loss
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--households", type=int, default=50)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_FORECAST_ARTIFACT)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    clients = household_ids(args.households)
    train_x, train_y, train_records = prepare_split(clients, pd.Timestamp("2025-01-08"), TRAIN_END)
    val_x, val_y, val_records = prepare_split(clients, TRAIN_END + pd.DateOffset(days=1), VALIDATION_END)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                              batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                            batch_size=args.batch_size, shuffle=False)
    source = str(CACHE_SNAPSHOT) if CACHE_SNAPSHOT.is_dir() else CHECKPOINT
    model = TinyTimeMixerForPrediction.from_pretrained(
        source, local_files_only=CACHE_SNAPSHOT.is_dir(), ignore_mismatched_sizes=True,
        context_length=336, prediction_length=48, patch_length=48, patch_stride=48,
        num_input_channels=1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.inference_mode(): validation_loss = run_epoch(model, val_loader, device)
        history.append({"epoch":epoch,"train_loss":train_loss,"validation_loss":validation_loss})
        print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.6f}, validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    artifact = args.artifact_dir
    model_dir = artifact / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    metadata = {
        "model_name":"Energy-TTM / TinyTimeMixerForPrediction",
        "model_version":"v1-30min-336x48",
        "source_notebooks":["energy_ttm_d1_forecasting.ipynb","energy_ttm_dr_backtesting_2025.ipynb"],
        "base_checkpoint":CHECKPOINT,
        "architecture":"TinyTimeMixerForPrediction",
        "adaptation":"patch_length and patch_stride 24->48; context 168->336; prediction head 24->48; all parameters fine-tuned",
        "context_length":336,"prediction_length":48,"frequency":"30min",
        "target":"aggregate_power_kw","energy_conversion":"kW * 0.5 h = kWh per interval",
        "normalization":"per-origin mean/std from the 336 historical values only",
        "scaler_artifact":None,
        "split":{"train":"2025-01-08/2025-09-30","validation":"2025-10-01/2025-10-31","test":"2025-11-01/2025-12-31"},
        "test_start":"2025-11-01","model_directory":"model","households":clients,
        "training_windows":len(train_records),"validation_windows":len(val_records),
        "epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate,
        "best_validation_loss":best_loss,"device":device,"random_seed":42,
        "trained_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(artifact / "training_history.csv", index=False)
    print(f"Saved artifact to {artifact}")


if __name__ == "__main__": main()
