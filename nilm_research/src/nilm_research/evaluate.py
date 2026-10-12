from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import save_json
from .data import NILMRepository, endpoint_quality, prepare_partition, quality_summary
from .metrics import MetricsAccumulator


@torch.inference_mode()
def predict_partition(model: torch.nn.Module, part, context: int, chunk_size: int,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(part.timestamps)
    power = np.full((n, 3), np.nan, dtype=np.float32)
    probability = np.full((n, 3), np.nan, dtype=np.float32)
    model.eval()
    for position in range(0, n, chunk_size):
        start = max(0, position - context + 1)
        stop = min(n, position + chunk_size)
        x = torch.from_numpy(part.inputs[:, start:stop]).unsqueeze(0).to(device)
        outputs = model.forward_sequence(x)
        local = position - start
        count = stop - position
        power[position:stop] = outputs["power_w"][0, local:local + count].cpu().numpy()
        probability[position:stop] = outputs["activity_probability"][0, local:local + count].cpu().numpy()
    endpoint, _ = endpoint_quality(part, context)
    power[~endpoint] = np.nan
    probability[~endpoint] = np.nan
    return power, probability, endpoint


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate_checkpoint(checkpoint: str | Path, splits: list[str] | None = None) -> dict:
    model, payload = load_checkpoint(checkpoint, "cpu")
    cfg, transforms = payload["config"], payload["transforms"]
    device_name = cfg["evaluation"].get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          ("cpu" if device_name == "auto" else device_name))
    model.to(device)
    repo = NILMRepository(cfg)
    panel = repo.panel_households()
    heldout = set(cfg["evaluation"].get("held_out_households", []))
    selected_splits = splits or ["validation", "test"]
    output = Path(checkpoint).resolve().parent
    all_rows, summaries = [], {}
    started = time.perf_counter()
    for split in selected_splits:
        accumulator = MetricsAccumulator(float(cfg["evaluation"]["activity_probability_threshold"]))
        parts = []
        split_start = time.perf_counter()
        for client in panel:
            part = prepare_partition(repo, cfg, transforms, client, split)
            parts.append(part)
            predicted, probability, endpoint = predict_partition(
                model, part, int(cfg["data"]["context_length"]),
                int(cfg["evaluation"]["sequence_chunk_size"]), device
            )
            accumulator.add_partition(part, predicted, probability, endpoint,
                                      "held_out_household" if client in heldout else "same_household")
            print(json.dumps({"evaluation_split": split, "household": client,
                              "valid_endpoints": int(endpoint.sum())}), flush=True)
        rows = accumulator.rows(split, payload["model_kind"])
        all_rows.extend(rows)
        summary = {"seconds": time.perf_counter() - split_start,
                   "quality": quality_summary(parts, int(cfg["data"]["context_length"])),
                   "households": panel, "held_out_households": sorted(heldout)}
        summaries[split] = summary
        _write_rows(output / f"metrics_{split}.csv", rows)
        save_json(output / f"evaluation_{split}.json", summary)
    overall = {"checkpoint": str(Path(checkpoint).resolve()), "device": str(device),
               "duration_seconds": time.perf_counter() - started, "splits": summaries}
    save_json(output / "evaluation_summary.json", overall)
    return overall
