#!/usr/bin/env python
import argparse
import json
import _bootstrap  # noqa: F401
from nilm_research.config import load_config, resolved_config, save_json
from nilm_research.data import NILMRepository, fit_transforms, prepare_partition, quality_summary

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--output")
args = p.parse_args()
cfg = resolved_config(load_config(args.config)); repo = NILMRepository(cfg)
verification = repo.verify(); heldout = set(cfg["evaluation"].get("held_out_households", []))
fit_homes = [x for x in verification["panel_households"] if x not in heldout]
transforms, provenance = fit_transforms(repo, cfg, fit_homes)
context = int(cfg["data"]["context_length"])
qualities = {}
for split in ("train", "validation", "test"):
    parts = [prepare_partition(repo, cfg, transforms, x, split) for x in verification["panel_households"]]
    qualities[split] = quality_summary(parts, context)
report = {"verification": verification, "fit_provenance": provenance, "transforms": transforms,
          "quality": qualities, "disclosure": "Two-sided imputation provenance recovered from cleaned parquet flags."}
if args.output:
    save_json(args.output, report)
print(json.dumps(report, indent=2, default=str))

