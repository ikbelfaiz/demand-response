#!/usr/bin/env python
import argparse
import json
import _bootstrap  # noqa: F401
from nilm_research.config import load_config, resolved_config
from nilm_research.train import run_training

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--model", choices=["tcn", "cnn_baseline"], required=True)
p.add_argument("--resume")
args = p.parse_args()
print(json.dumps(run_training(resolved_config(load_config(args.config)), args.model, args.resume), indent=2))

