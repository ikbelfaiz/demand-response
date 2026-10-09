#!/usr/bin/env python
import argparse
from pathlib import Path
import subprocess
import sys
import _bootstrap  # noqa: F401
from nilm_research.config import load_config, resolved_config

p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
args = p.parse_args(); here = Path(__file__).resolve().parent
cfg = resolved_config(load_config(args.config)); output_root = Path(cfg["run"]["output_dir"])
for model in ("tcn", "cnn_baseline"):
    subprocess.run([sys.executable, str(here / "train.py"), "--config", args.config, "--model", model], check=True)
    checkpoint = output_root / model / "best.pt"
    subprocess.run([sys.executable, str(here / "evaluate.py"), "--checkpoint", str(checkpoint)], check=True)
    subprocess.run([sys.executable, str(here / "plot_results.py"), "--run-dir", str(checkpoint.parent)], check=True)
subprocess.run([sys.executable, str(here / "summarize_execution.py"), "--root", str(output_root)], check=True)
