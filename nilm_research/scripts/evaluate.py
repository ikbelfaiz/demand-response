#!/usr/bin/env python
import argparse
import json
import _bootstrap  # noqa: F401
from nilm_research.evaluate import evaluate_checkpoint

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--splits", nargs="*", choices=["validation", "test"])
args = p.parse_args()
print(json.dumps(evaluate_checkpoint(args.checkpoint, args.splits), indent=2))

