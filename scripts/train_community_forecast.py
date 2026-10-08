"""Train causal profile models for community demand, STEG, and PV forecasts."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from backend.data import load_events
from backend.intelligence.community_forecasting import (
    COMMUNITY_ARTIFACT, MODEL_COLUMNS, ProfileModel, build_training_matrix,
    load_causal_operational_grid,
)
from backend.intelligence.forecast_metrics import forecast_metrics

TRAIN_DATES = pd.date_range("2025-01-09", "2025-05-31", freq="1D")
VALIDATION_DATES = pd.date_range("2025-06-01", "2025-06-30", freq="1D")
TEST_DATES = pd.date_range("2025-07-01", "2025-08-31", freq="1D")
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def intervention_affected_dates() -> set[pd.Timestamp]:
    events = load_events()
    affected = set()
    for row in events.itertuples():
        start, end = pd.Timestamp(row.start_ts), pd.Timestamp(row.end_ts)
        affected.update(pd.date_range(start.normalize(), end.normalize(), freq="1D"))
    return affected


def main() -> None:
    frame = load_causal_operational_grid()
    affected = intervention_affected_dates()
    models, selection_rows, counts = {}, [], {}
    for column in MODEL_COLUMNS:
        excluded = affected if column == "households_consumption_kw" else frozenset()
        train_x, train_y, train_days = build_training_matrix(frame, TRAIN_DATES, column, excluded_dates=excluded)
        val_x, val_y, val_days = build_training_matrix(frame, VALIDATION_DATES, column, excluded_dates=excluded)
        scaler = StandardScaler().fit(train_x)
        scaled_train, scaled_val = scaler.transform(train_x), scaler.transform(val_x)
        best = None
        for alpha in ALPHAS:
            regressor = Ridge(alpha=alpha).fit(scaled_train, train_y)
            prediction = np.clip(regressor.predict(scaled_val), 0, None)
            metrics = forecast_metrics(val_y.ravel(), prediction.ravel())
            # Energy and peak statistics are daily quantities.  Computing them
            # on the flattened validation matrix creates artificial multi-day
            # peaks and meaningless timing offsets, although slot MAE/RMSE/WAPE
            # remain valid when flattened.
            daily = pd.DataFrame([
                forecast_metrics(actual, predicted)
                for actual, predicted in zip(val_y, prediction)
            ]).mean(numeric_only=True)
            for key in daily.index:
                if key.startswith("daily_energy_") or key.startswith("peak_"):
                    metrics[key] = float(daily[key])
            selection_rows.append({"column":column,"alpha":alpha,"validation_days":len(val_days),**metrics})
            candidate = (metrics["rmse_kw"], alpha, regressor)
            if best is None or candidate[0] < best[0]: best = candidate
        models[column] = ProfileModel(column, scaler, best[2])
        counts[column] = {"training_windows":len(train_days),"validation_windows":len(val_days),
                          "selected_alpha":best[1],"validation_rmse_kw":best[0]}
        print(column, counts[column], flush=True)
    COMMUNITY_ARTIFACT.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, COMMUNITY_ARTIFACT / "models.joblib")
    pd.DataFrame(selection_rows).to_csv(COMMUNITY_ARTIFACT / "model_selection.csv", index=False)
    metadata = {
        "model_name":"Causal multi-output Ridge profile forecaster",
        "model_version":"community-profile-ridge-v1",
        "forecast_slots":48,"frequency":"30min","issuance_hour":14,
        "targets":{"households_consumption_kw":"kW","steg_production_kw":"kW","pv_production_kw":"kW"},
        "features":["48 half-hours ending D-1 13:30","D-2 profile","D-7 profile",
                    "mean profile D-8 through D-2","target day-of-week one-hot",
                    "target month sine/cosine","known holiday flag","known Ramadan flag"],
        "weather_features":"omitted: no archived day-ahead weather forecast exists",
        "source":"original pre-completion grid; >=80% source-minute coverage per slot; no two-sided interpolation",
        "normalization":"StandardScaler fitted only on the training split for each target",
        "split":{"train":"2025-01-09/2025-05-31","validation":"2025-06-01/2025-06-30",
                 "test":"2025-07-01/2025-08-31"},
        "intervention_strategy":"Historical intervention target days excluded from demand training/validation; supply models retain them.",
        "counts":counts,"alpha_candidates":list(ALPHAS),"randomness":"deterministic closed-form Ridge",
        "trained_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    (COMMUNITY_ARTIFACT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {COMMUNITY_ARTIFACT}")


if __name__ == "__main__": main()
