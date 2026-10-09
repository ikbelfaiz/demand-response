#!/usr/bin/env python
import argparse
import csv
import json
import platform
from pathlib import Path
import torch

p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
args = p.parse_args(); root = Path(args.root)
models = ("tcn", "cnn_baseline"); splits = ("validation", "test")
summary = {"hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                         "torch_threads": torch.get_num_threads()}, "models": {}}
for model in models:
    history = list(csv.DictReader((root / model / "history.csv").open()))
    epoch_seconds = sum(float(x["seconds"]) for x in history)
    item = {"epoch_compute_seconds_total": epoch_seconds,
            "epochs": len(history), "best_validation_loss": min(float(x["validation_loss"]) for x in history),
            "splits": {}}
    for split in splits:
        rows = list(csv.DictReader((root / model / f"metrics_{split}.csv").open()))
        item["splits"][split] = {
            r["appliance"]: {k: (None if r[k] == "" else float(r[k])) for k in
                             ("power_mae_w", "hourly_energy_mae_kwh", "relative_total_energy_error",
                              "activity_precision", "activity_recall", "activity_f1", "mean_hourly_coverage")}
            | {"valid_observations": int(r["valid_observations"])}
            for r in rows if r["household"] == "ALL" and r["stratum"] == "overall"
        }
    summary["models"][model] = item
    training_path = root / model / "training_summary.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if "duration_seconds" in training:
        training["duration_seconds_current_invocation"] = training.pop("duration_seconds")
    training["epoch_compute_seconds_total"] = epoch_seconds
    training_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
summary["test_tcn_minus_baseline"] = {
    appliance: {
        metric: summary["models"]["tcn"]["splits"]["test"][appliance][metric]
                - summary["models"]["cnn_baseline"]["splits"]["test"][appliance][metric]
        for metric in ("power_mae_w", "hourly_energy_mae_kwh", "activity_f1")
    } for appliance in ("ac", "water_heater", "washing_machine")
}
(root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
