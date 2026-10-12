#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import _bootstrap 
from nilm_research.inference import run_inference
from nilm_research.plotting import plot_inference

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True); p.add_argument("--household", required=True)
p.add_argument("--start", required=True); p.add_argument("--end", required=True)
p.add_argument("--output", required=True); p.add_argument("--device", default="auto")
p.add_argument("--plot")
args = p.parse_args()
result = run_inference(args.checkpoint, args.household, args.start, args.end, args.output, args.device)
if args.plot:
    plot_inference(args.output, args.plot); result["plot"] = str(Path(args.plot).resolve())
print(json.dumps(result, indent=2))

