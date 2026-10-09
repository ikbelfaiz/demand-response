from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .checkpoint import load_checkpoint, save_checkpoint
from .config import save_json
from .data import NILMRepository, WindowDataset, fit_transforms, prepare_partition, quality_summary
from .losses import multitask_loss
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(cfg: dict) -> torch.device:
    requested = cfg["training"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def _limited_subset(dataset: WindowDataset, maximum: int) -> DatasetLike:
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64).tolist())


DatasetLike = WindowDataset | Subset


def _epoch(model, loader, transforms, loss_cfg, device, optimizer=None, scaler=None) -> dict:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_batches = 0
    amp_enabled = bool(scaler is not None and scaler.is_enabled())
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(batch["inputs"])
            loss, pieces = multitask_loss(outputs, batch, transforms, loss_cfg)
        if training:
            if amp_enabled:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        for key, value in pieces.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()} | {"batches": n_batches}


def run_training(cfg: dict, model_kind: str, resume: str | None = None) -> dict:
    seed_everything(int(cfg["training"]["seed"]))
    device = _device(cfg)
    output = Path(cfg["run"]["output_dir"]) / model_kind
    output.mkdir(parents=True, exist_ok=True)
    repo = NILMRepository(cfg)
    panel = repo.panel_households()
    heldout = sorted(set(cfg["evaluation"].get("held_out_households", [])))
    unknown = sorted(set(heldout) - set(panel))
    if unknown:
        raise ValueError(f"Held-out households are not labeled panel homes: {unknown}")
    fit_homes = [x for x in panel if x not in heldout]
    if not fit_homes:
        raise ValueError("No training households remain")
    transforms, fit_record = fit_transforms(repo, cfg, fit_homes)
    save_json(output / "transforms.json", transforms)
    save_json(output / "fit_provenance.json", fit_record)

    train_parts = [prepare_partition(repo, cfg, transforms, x, "train") for x in fit_homes]
    val_parts = [prepare_partition(repo, cfg, transforms, x, "validation") for x in fit_homes]
    context = int(cfg["data"]["context_length"])
    train_data = WindowDataset(train_parts, context)
    val_data = WindowDataset(val_parts, context)
    save_json(output / "quality_train.json", quality_summary(train_parts, context))
    save_json(output / "quality_validation.json", quality_summary(val_parts, context))

    weights = train_data.sampling_weights(float(cfg["sampling"]["active_weight"]),
                                          float(cfg["sampling"]["transition_weight"]))
    samples = min(int(cfg["training"]["samples_per_epoch"]), max(len(train_data), 1))
    sampler = WeightedRandomSampler(weights, samples, replacement=True,
                                    generator=torch.Generator().manual_seed(int(cfg["training"]["seed"])))
    workers = int(cfg["training"].get("workers", 0))
    train_loader = DataLoader(train_data, batch_size=int(cfg["training"]["batch_size"]), sampler=sampler,
                              num_workers=workers, pin_memory=device.type == "cuda")
    val_subset = _limited_subset(val_data, int(cfg["training"].get("validation_windows", 0)))
    val_loader = DataLoader(val_subset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda")

    model = build_model(model_kind, transforms, cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    start_epoch, best, history = 0, float("inf"), []
    if resume:
        resumed, payload = load_checkpoint(resume, device)
        if payload["model_kind"] != model_kind:
            raise ValueError("Resume checkpoint model kind mismatch")
        model.load_state_dict(resumed.state_dict())
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best = float(payload["best_validation_loss"])
        history = list(payload.get("history", []))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(cfg["training"].get("mixed_precision", True)))
    patience, stale = int(cfg["training"]["early_stopping_patience"]), 0
    started = time.perf_counter()
    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        tick = time.perf_counter()
        train_metrics = _epoch(model, train_loader, transforms, cfg["loss"], device, optimizer, scaler)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, transforms, cfg["loss"], device)
        record = {"epoch": epoch, "seconds": time.perf_counter() - tick,
                  **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"validation_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        save_checkpoint(output / "last.pt", model, optimizer, epoch, best, cfg, transforms, model_kind, history)
        if val_metrics.get("loss", float("inf")) < best:
            best = val_metrics["loss"]; stale = 0
            save_checkpoint(output / "best.pt", model, optimizer, epoch, best, cfg, transforms, model_kind, history)
        else:
            stale += 1
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= patience:
            break
    duration = time.perf_counter() - started
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in history for k in row}))
        writer.writeheader(); writer.writerows(history)
    summary = {"model_kind": model_kind, "device": str(device),
               "duration_seconds_current_invocation": duration,
               "epoch_compute_seconds_total": sum(float(x.get("seconds", 0.0)) for x in history),
               "best_validation_loss": best, "epochs_completed": len(history), "fit_households": fit_homes,
               "held_out_households": heldout, "train_windows": len(train_data), "validation_windows": len(val_data),
               "checkpoint": str(output / "best.pt")}
    save_json(output / "training_summary.json", summary)
    return summary
