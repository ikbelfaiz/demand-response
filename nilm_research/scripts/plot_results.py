#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from nilm_research.plotting import plot_history, plot_household_errors

p = argparse.ArgumentParser(); p.add_argument("--run-dir", required=True)
args = p.parse_args(); root = Path(args.run_dir)
plot_history(root / "history.csv", root / "loss_curves.svg")
for split in ("validation", "test"):
    source = root / f"metrics_{split}.csv"
    if source.exists(): plot_household_errors(source, root / f"errors_{split}.svg")
