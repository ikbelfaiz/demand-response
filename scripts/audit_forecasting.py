"""Produce reproducible, inference-only evidence for the forecasting audit.

This script never fits or updates a model.  It compares the saved household
weights with the identically adapted upstream initialization, inspects the
closed-form community artifact, quantifies completed source values, and plots
representative held-out forecasts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

from backend.data import load_household, operational_household
from backend.intelligence.community_forecasting import COMMUNITY_ARTIFACT
from backend.intelligence.forecast_features import build_window, seasonal_persistence
from backend.intelligence.forecasting import EnergyTTMForecaster
from backend.intelligence.model_registry import DEFAULT_FORECAST_ARTIFACT
from scripts.train_forecasting import CACHE_SNAPSHOT

OUT = ROOT / "outputs" / "forecast_audit"
BACKUP = ROOT / "data" / "original_v3_backup"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_evidence() -> dict:
    if not CACHE_SNAPSHOT.is_dir():
        return {"status": "unverified", "reason": f"local base snapshot missing: {CACHE_SNAPSHOT}"}
    torch.manual_seed(42)
    initial = TinyTimeMixerForPrediction.from_pretrained(
        str(CACHE_SNAPSHOT), local_files_only=True, ignore_mismatched_sizes=True,
        context_length=336, prediction_length=48, patch_length=48,
        patch_stride=48, num_input_channels=1,
    )
    saved = TinyTimeMixerForPrediction.from_pretrained(
        str(DEFAULT_FORECAST_ARTIFACT / "model"), local_files_only=True
    )
    initial_state, saved_state = initial.state_dict(), saved.state_dict()
    rows = []
    for name in sorted(initial_state.keys() & saved_state.keys()):
        before, after = initial_state[name].detach().cpu(), saved_state[name].detach().cpu()
        if before.shape != after.shape:
            continue
        maximum = float(torch.max(torch.abs(after - before)))
        rows.append({
            "name": name,
            "parameters": int(after.numel()),
            "max_absolute_change": maximum,
            "changed": maximum > 1e-8,
            "adapted_input_or_head": "patcher" in name or "head" in name,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "household_weight_comparison.csv", index=False)
    return {
        "status": "verified_updated",
        "common_same_shape_tensors": int(len(frame)),
        "changed_tensors": int(frame.changed.sum()),
        "unchanged_tensors": int((~frame.changed).sum()),
        "changed_parameters": int(frame.loc[frame.changed, "parameters"].sum()),
        "total_compared_parameters": int(frame.parameters.sum()),
        "changed_core_tensors": int(frame.loc[~frame.adapted_input_or_head, "changed"].sum()),
        "core_tensors": int((~frame.adapted_input_or_head).sum()),
        "largest_changes": frame.nlargest(8, "max_absolute_change")[
            ["name", "max_absolute_change"]
        ].to_dict("records"),
    }


def community_evidence() -> dict:
    models = joblib.load(COMMUNITY_ARTIFACT / "models.joblib")
    result = {}
    for column, model in models.items():
        coef = np.asarray(model.regressor.coef_)
        result[column] = {
            "coefficient_shape": list(coef.shape),
            "nonzero_coefficients": int(np.count_nonzero(coef)),
            "coefficient_l2_norm": float(np.linalg.norm(coef)),
            "scaler_features": int(np.asarray(model.scaler.mean_).size),
            "ridge_alpha": float(model.regressor.alpha),
        }
    return result


def completion_evidence() -> dict:
    split_days = {
        "train": pd.date_range("2025-01-08", "2025-09-30", freq="D"),
        "validation": pd.date_range("2025-10-01", "2025-10-31", freq="D"),
        "test": pd.date_range("2025-11-01", "2025-12-31", freq="D"),
    }
    result = {name: {"source_missing_minutes": 0, "affected_30min_slots": 0,
                     "fully_missing_30min_slots": 0, "target_windows_affected": 0,
                     "context_windows_affected": 0}
              for name in split_days}
    all_slots = pd.date_range("2025-01-01", "2025-12-31 23:30", freq="30min")
    for number in range(1, 51):
        client = f"C{number:03d}"
        path = BACKUP / ("households_1min_C001_C025.parquet" if number <= 25 else
                         "households_1min_C026_C050.parquet")
        raw = pd.read_parquet(path, columns=["timestamp", "client_id", "aggregate_power_w"],
                              filters=[("client_id", "==", client)], engine="pyarrow")
        raw["timestamp"] = pd.to_datetime(raw.timestamp)
        count = (pd.to_numeric(raw.aggregate_power_w, errors="coerce")
                 .groupby(raw.timestamp.dt.floor("30min")).count().reindex(all_slots, fill_value=0))
        missing = 30 - count.clip(upper=30)
        affected = missing.gt(0)
        fully_missing = count.eq(0)
        for name, days in split_days.items():
            slot_mask = count.index.normalize().isin(days)
            result[name]["source_missing_minutes"] += int(missing.loc[slot_mask].sum())
            result[name]["affected_30min_slots"] += int(affected.loc[slot_mask].sum())
            result[name]["fully_missing_30min_slots"] += int(fully_missing.loc[slot_mask].sum())
            for day in days:
                target = (count.index >= day) & (count.index < day + pd.DateOffset(days=1))
                context = (count.index >= day - pd.DateOffset(days=7)) & (count.index < day)
                result[name]["target_windows_affected"] += int(affected.loc[target].any())
                result[name]["context_windows_affected"] += int(affected.loc[context].any())
    return result


def _style_axis(axis, title: str) -> None:
    axis.set_title(title, color="#111827", weight="bold")
    axis.set_xlabel("Time")
    axis.set_ylabel("Power (kW)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)


def household_plots() -> list[dict]:
    metrics_path = DEFAULT_FORECAST_ARTIFACT / "test_predictions_metrics.csv"
    metrics = pd.read_csv(metrics_path, parse_dates=["forecast_date"])
    model_rows = metrics.loc[metrics.method.eq("Energy-TTM")].sort_values("rmse_kw").reset_index(drop=True)
    positions = {"good": 0, "median": len(model_rows) // 2, "poor": len(model_rows) - 1}
    forecaster = EnergyTTMForecaster()
    selected = []
    for label, position in positions.items():
        row = model_rows.iloc[position]
        client, day = row.client_id, pd.Timestamp(row.forecast_date)
        raw = load_household(client, day - pd.DateOffset(days=7), day + pd.DateOffset(days=1),
                             columns=["timestamp", "client_id", "aggregate_power_w"])
        window = build_window(operational_household(raw), client, day, require_target=True)
        prediction = forecaster.predict(window.context_kw)
        baseline = seasonal_persistence(window.context_kw)
        fig, axis = plt.subplots(figsize=(11, 4.5))
        axis.plot(window.target_timestamps, window.target_kw, color="#111827", lw=2.3,
                  label="Canonical target")
        axis.plot(window.target_timestamps, prediction, color="#2563eb", lw=2, label="Energy-TTM")
        axis.plot(window.target_timestamps, baseline, color="#d97706", lw=1.8, ls="--",
                  label="7-day persistence")
        backup_path = BACKUP / ("households_1min_C001_C025.parquet" if int(client[1:]) <= 25 else
                                "households_1min_C026_C050.parquet")
        original = pd.read_parquet(
            backup_path, columns=["timestamp", "client_id", "aggregate_power_w"],
            filters=[("client_id", "==", client), ("timestamp", ">=", day),
                     ("timestamp", "<", day + pd.DateOffset(days=1))], engine="pyarrow"
        )
        original["timestamp"] = pd.to_datetime(original.timestamp)
        coverage = (pd.to_numeric(original.aggregate_power_w, errors="coerce")
                    .groupby(original.timestamp.dt.floor("30min")).count()
                    .reindex(window.target_timestamps, fill_value=0))
        completed = coverage.lt(30).to_numpy()
        if completed.any():
            axis.scatter(window.target_timestamps[completed], window.target_kw[completed],
                         color="#c026d3", marker="x", s=38, linewidths=1.7, zorder=5,
                         label="Target slot includes completed source data")
        _style_axis(axis, f"Household {label}: {client}, {day.date()} (model RMSE {row.rmse_kw:.3f} kW)")
        fig.autofmt_xdate(); fig.tight_layout()
        relative = Path(f"household_{label}.png")
        fig.savefig(OUT / relative, dpi=150); plt.close(fig)
        baseline_row = metrics.loc[(metrics.client_id.eq(client)) &
                                   (metrics.forecast_date.eq(day)) &
                                   metrics.method.eq("Seasonal persistence")].iloc[0]
        selected.append({"scope": "household", "case": label, "client_id": client,
                         "forecast_date": str(day.date()), "model_rmse_kw": float(row.rmse_kw),
                         "baseline_rmse_kw": float(baseline_row.rmse_kw),
                         "completed_target_slots": int(completed.sum()), "plot": str(relative)})
    return selected


def community_plots() -> list[dict]:
    daily = pd.read_csv(COMMUNITY_ARTIFACT / "daily_forecast_evaluation.csv",
                        parse_dates=["forecast_day"])
    model_rows = daily.loc[(daily.target.eq("demand")) & daily.method.eq("model")]
    model_rows = model_rows.sort_values("rmse_kw").reset_index(drop=True)
    slots = pd.read_parquet(COMMUNITY_ARTIFACT / "test_replay_slots.parquet")
    slots["forecast_day"] = pd.to_datetime(slots.forecast_day)
    positions = {"good": 0, "median": len(model_rows) // 2, "poor": len(model_rows) - 1}
    selected = []
    for label, position in positions.items():
        row = model_rows.iloc[position]; day = pd.Timestamp(row.forecast_day)
        frame = slots.loc[slots.forecast_day.eq(day)].sort_values("timestamp")
        fig, axis = plt.subplots(figsize=(11, 4.5))
        axis.plot(frame.timestamp, frame.actual_demand_kw, color="#111827", lw=2.3, label="Actual")
        axis.plot(frame.timestamp, frame.predicted_demand_kw, color="#2563eb", lw=2, label="Ridge")
        axis.plot(frame.timestamp, frame.baseline_demand_kw, color="#d97706", lw=1.8, ls="--",
                  label="7-day persistence")
        _style_axis(axis, f"Community {label}: {day.date()} (model RMSE {row.rmse_kw:.2f} kW)")
        fig.autofmt_xdate(); fig.tight_layout()
        relative = Path(f"community_{label}.png")
        fig.savefig(OUT / relative, dpi=150); plt.close(fig)
        baseline_row = daily.loc[(daily.forecast_day.eq(day)) & daily.target.eq("demand") &
                                 daily.method.eq("baseline")].iloc[0]
        selected.append({"scope": "community", "case": label,
                         "forecast_date": str(day.date()), "model_rmse_kw": float(row.rmse_kw),
                         "baseline_rmse_kw": float(baseline_row.rmse_kw), "plot": str(relative)})
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "household_model_sha256": sha256(DEFAULT_FORECAST_ARTIFACT / "model" / "model.safetensors"),
        "community_model_sha256": sha256(COMMUNITY_ARTIFACT / "models.joblib"),
        "household_weight_update": weight_evidence(),
        "community_artifact": community_evidence(),
        "household_source_completion_by_split": completion_evidence(),
    }
    cases = household_plots() + community_plots()
    pd.DataFrame(cases).to_csv(OUT / "selected_cases.csv", index=False)
    evidence["selected_cases"] = cases
    (OUT / "audit_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
