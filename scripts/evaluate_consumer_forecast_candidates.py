"""Select consumer candidates on validation, then evaluate one model on untouched test data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backend.intelligence.consumer_forecast_data import build_consumer_window, load_causal_household
from backend.intelligence.forecast_metrics import forecast_metrics
from backend.intelligence.forecasting import EnergyTTMForecaster
from backend.intelligence.model_registry import DEFAULT_FORECAST_ARTIFACT

OUT = ROOT / "outputs" / "consumer_forecast_improvement"
CANDIDATES = ROOT / "models" / "household_forecasting" / "candidates"
VALIDATION_DATES = pd.date_range("2025-10-01", "2025-10-31", freq="D")
TEST_DATES = pd.date_range("2025-11-01", "2025-12-31", freq="D")
ENSEMBLES = {
    "Energy-TTM peak-calibrated": {"Current Energy-TTM": 0.75, "energy_ttm_causal_peak": 0.25},
    "Energy-TTM profile-calibrated": {"Current Energy-TTM": 0.75, "Weekday/weekend profile": 0.25},
    "Energy-TTM balanced calibration": {"Current Energy-TTM": 0.50,
                                         "energy_ttm_causal_peak": 0.25,
                                         "Weekday/weekend profile": 0.25},
}


def profile_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    work = frame[["timestamp", column]].copy()
    work["date"] = work.timestamp.dt.normalize()
    work["slot"] = work.timestamp.dt.hour * 2 + work.timestamp.dt.minute // 30
    return work.pivot(index="date", columns="slot", values=column).reindex(columns=range(48))


def baselines(profiles: pd.DataFrame, day: pd.Timestamp) -> dict[str, np.ndarray]:
    prior = profiles.loc[profiles.index < day]
    same_type = prior.loc[(prior.index.dayofweek >= 5) == (day.dayofweek >= 5)]
    return {
        "Previous-day persistence": profiles.loc[day - pd.DateOffset(days=1)].to_numpy(np.float32),
        "Previous-week persistence": profiles.loc[day - pd.DateOffset(days=7)].to_numpy(np.float32),
        "Historical slot median": prior.median(axis=0).to_numpy(np.float32),
        "Weekday/weekend profile": same_type.median(axis=0).to_numpy(np.float32),
    }


def scored(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = forecast_metrics(actual, prediction)
    peak = np.argpartition(actual, -6)[-6:]
    error = prediction[peak] - actual[peak]
    result["peak_interval_mae_kw"] = float(np.abs(error).mean())
    result["peak_interval_rmse_kw"] = float(np.sqrt(np.mean(error ** 2)))
    return result


def evaluate(dates: pd.DatetimeIndex, models: dict[str, EnergyTTMForecaster],
             *, output_learned: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for number in range(1, 51):
        client = f"C{number:03d}"; frame = load_causal_household(client)
        profiles = profile_table(frame, "context_power_kw")
        for day in dates:
            try:
                window = build_consumer_window(frame, client, day, require_target=True)
            except ValueError:
                continue
            methods = baselines(profiles, day)
            methods.update({name: model.predict(window.context_kw) for name, model in models.items()})
            for name, weights in ENSEMBLES.items():
                methods[name] = sum(weight * methods[component] for component, weight in weights.items())
            if output_learned is not None:
                methods = {name: values for name, values in methods.items()
                           if name in output_learned or name not in {*models, *ENSEMBLES}}
            for name, prediction in methods.items():
                rows.append({"client_id": client, "forecast_date": day, "method": name,
                             **scored(window.target_kw, np.asarray(prediction))})
        print(f"Evaluated {client} ({number}/50)", flush=True)
    return pd.DataFrame(rows)


def summaries(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    metric_columns = [column for column in frame if column not in {"client_id", "forecast_date", "method"}]
    overall = frame.groupby("method", as_index=False)[metric_columns].mean()
    per_household = frame.groupby(["client_id", "method"], as_index=False)[metric_columns].mean()
    per_month = (frame.assign(month=frame.forecast_date.dt.to_period("M").astype(str))
                 .groupby(["month", "method"], as_index=False)[metric_columns].mean())
    frame.to_csv(OUT / f"{prefix}_daily.csv", index=False)
    overall.to_csv(OUT / f"{prefix}_overall.csv", index=False)
    per_household.to_csv(OUT / f"{prefix}_per_household.csv", index=False)
    per_month.to_csv(OUT / f"{prefix}_per_month.csv", index=False)
    return overall


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current = EnergyTTMForecaster(DEFAULT_FORECAST_ARTIFACT)
    candidates = {
        path.name: EnergyTTMForecaster(path)
        for path in sorted(CANDIDATES.iterdir())
        if (path / "metadata.json").is_file()
    }
    validation = evaluate(VALIDATION_DATES, {"Current Energy-TTM": current, **candidates})
    validation_overall = summaries(validation, "validation")
    learned_names = ["Current Energy-TTM", *candidates, *ENSEMBLES]
    learned = validation_overall.loc[validation_overall.method.isin(learned_names)]
    selected_name = learned.sort_values("rmse_kw").iloc[0].method
    current_row = learned.loc[learned.method.eq("Current Energy-TTM")].iloc[0]
    selected_row = learned.loc[learned.method.eq(selected_name)].iloc[0]
    candidate_wins = selected_name != "Current Energy-TTM"
    meaningful = bool(candidate_wins and
                      selected_row.rmse_kw <= current_row.rmse_kw * 0.98 and
                      selected_row.mae_kw <= current_row.mae_kw * 0.98 and
                      selected_row.wape_pct <= current_row.wape_pct and
                      selected_row.peak_interval_rmse_kw <= current_row.peak_interval_rmse_kw)
    active_name = selected_name if meaningful else "Current Energy-TTM"
    selected_model = candidates[active_name] if active_name in candidates else current
    # The test period is first opened only after validation has fixed active_name.
    test_models = {"Current Energy-TTM": current, **candidates}
    test = evaluate(TEST_DATES, test_models,
                    output_learned={"Current Energy-TTM", active_name})
    test_overall = summaries(test, "test_stage2")
    selection = {
        "selection_period": "2025-10-01/2025-10-31",
        "untouched_test_period": "2025-11-01/2025-12-31",
        "best_validation_method": selected_name,
        "meaningful_improvement_rule": "at least 2% lower validation RMSE and MAE, no worse WAPE or peak-interval RMSE",
        "candidate_passed": meaningful,
        "selected_for_consumer": active_name,
        "validation_windows_per_method": int(validation.groupby("method").size().min()),
        "test_windows_per_method": int(test.groupby("method").size().min()),
        "validation_metrics": validation_overall.to_dict("records"),
        "test_metrics": test_overall.to_dict("records"),
    }
    selection["predeclared_calibrations"] = ENSEMBLES
    (OUT / "selection_stage2.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
