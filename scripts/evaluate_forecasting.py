"""Evaluate saved Energy-TTM and seven-day seasonal persistence on held-out data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backend.data import load_household, operational_household
from backend.intelligence.forecast_features import build_window, seasonal_persistence
from backend.intelligence.forecast_metrics import forecast_metrics
from backend.intelligence.forecasting import EnergyTTMForecaster
from backend.intelligence.model_registry import DEFAULT_FORECAST_ARTIFACT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_FORECAST_ARTIFACT)
    parser.add_argument("--households", type=int, default=50)
    args = parser.parse_args()
    model = EnergyTTMForecaster(args.artifact_dir)
    rows = []
    for number in range(1, args.households + 1):
        client_id = f"C{number:03d}"
        raw = load_household(client_id, "2025-10-25", "2026-01-01",
                             columns=["timestamp","client_id","aggregate_power_w"])
        operational = operational_household(raw)
        for day in pd.date_range("2025-11-01", "2025-12-31", freq="1D"):
            window = build_window(operational, client_id, day, require_target=True)
            for method, prediction in (("Energy-TTM", model.predict(window.context_kw)),
                                       ("Seasonal persistence", seasonal_persistence(window.context_kw))):
                rows.append({"client_id":client_id,"forecast_date":day,"method":method,
                             **forecast_metrics(window.target_kw, prediction)})
        print(f"Evaluated {client_id} ({number}/{args.households})", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.artifact_dir / "test_predictions_metrics.csv", index=False)
    metrics = ["mae_kw","rmse_kw","wape_pct","daily_energy_error_kwh",
               "peak_magnitude_error_kw","peak_timing_error_minutes"]
    overall = results.groupby("method", as_index=False)[metrics].mean()
    per_household = results.groupby(["client_id","method"], as_index=False)[metrics].mean()
    per_month = results.assign(month=results.forecast_date.dt.to_period("M").astype(str)).groupby(
        ["month","method"], as_index=False)[metrics].mean()
    overall.to_csv(args.artifact_dir / "evaluation_overall.csv", index=False)
    per_household.to_csv(args.artifact_dir / "evaluation_per_household.csv", index=False)
    per_month.to_csv(args.artifact_dir / "evaluation_per_month.csv", index=False)
    summary = {"test_start":"2025-11-01","test_end":"2025-12-31",
               "households":args.households,"test_windows":args.households*61,
               "overall":overall.to_dict(orient="records")}
    (args.artifact_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata_path = args.artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evaluation"] = summary
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__": main()
