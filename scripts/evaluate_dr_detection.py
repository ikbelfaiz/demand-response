"""Walk-forward held-out evaluation of community forecasting and DR recommendations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backend.data import load_events, load_grid, load_ground_truth_for_evaluation, operational_grid
from backend.intelligence.backtesting import classification_metrics, event_matching_metrics, mark_windows
from backend.intelligence.community_forecasting import (
    COMMUNITY_ARTIFACT, CommunityForecastBundle, build_target, forecast_index,
    issue_timestamp, load_causal_operational_grid,
)
from backend.intelligence.decision_engine import DecisionPolicy, policy_metadata, recommend_events
from backend.intelligence.forecast_metrics import forecast_metrics
from backend.intelligence.peak_detection import boolean_windows, detect_risk_windows

TEST_DATES = pd.date_range("2025-07-01", "2025-08-31", freq="1D")


def main() -> None:
    bundle = CommunityForecastBundle()
    causal = load_causal_operational_grid()
    actual_all = operational_grid(load_grid()).set_index("timestamp")
    policy = DecisionPolicy()  # fixed research thresholds; validation produced no predicted deficits
    slots, recommendations, baseline_windows = [], [], []
    daily_forecast_metrics = []
    issue_times = {}
    for day in TEST_DATES:
        forecast = bundle.predict_day(causal, day)
        index = forecast_index(day); actual = actual_all.reindex(index).reset_index()
        causal_actual = causal.set_index("timestamp").reindex(index)
        combined = forecast.copy()
        combined["actual_demand_kw"] = causal_actual.households_consumption_kw.to_numpy()
        combined["actual_steg_kw"] = causal_actual.steg_production_kw.to_numpy()
        combined["actual_pv_kw"] = causal_actual.pv_production_kw.to_numpy()
        combined["actual_supply_kw"] = (causal_actual.steg_production_kw+causal_actual.pv_production_kw).to_numpy()
        combined["actual_peak"] = actual.is_dr_peak.fillna(0).astype(int)
        combined["actual_event"] = actual.is_dr_event.fillna(0).astype(int)
        combined["evaluation_valid_peak"] = ((causal_actual.households_consumption_kw_coverage.eq(1) &
            causal_actual.pv_production_kw_coverage.eq(1) & causal_actual.is_dr_peak_coverage.eq(1)).to_numpy())
        combined["forecast_day"] = day
        baseline = {}
        for source, name in (("households_consumption_kw","demand"),("steg_production_kw","steg"),("pv_production_kw","pv")):
            baseline[name] = build_target(causal, day-pd.DateOffset(days=7), source)
        baseline["supply"] = baseline["steg"] + baseline["pv"]
        combined["baseline_demand_kw"] = baseline["demand"]
        combined["baseline_steg_kw"] = baseline["steg"]
        combined["baseline_pv_kw"] = baseline["pv"]
        combined["baseline_supply_kw"] = baseline["supply"]
        combined["baseline_margin_kw"] = baseline["supply"] - baseline["demand"]
        for name, actual_col, prediction_col in (
            ("demand","actual_demand_kw","predicted_demand_kw"),("steg","actual_steg_kw","predicted_steg_kw"),
            ("pv","actual_pv_kw","predicted_pv_kw"),("supply","actual_supply_kw","predicted_supply_kw")):
            for method, values in (("model",combined[prediction_col]),("baseline",baseline[name])):
                metrics = forecast_metrics(combined[actual_col], values)
                daily_forecast_metrics.append({"forecast_day":day,"target":name,"method":method,**metrics})
        risks = detect_risk_windows(forecast, safety_margin_kw=policy.safety_margin_kw)
        proposed = recommend_events(forecast, risks, issue_time=issue_timestamp(day),
                                    model_version=bundle.metadata["model_version"], policy=policy)
        if not proposed.empty: recommendations.append(proposed)
        baseline_forecast = forecast[["timestamp"]].copy()
        baseline_forecast["predicted_demand_kw"] = baseline["demand"]
        baseline_forecast["predicted_supply_kw"] = baseline["supply"]
        baseline_forecast["predicted_margin_kw"] = combined.baseline_margin_kw
        baseline_risks = detect_risk_windows(baseline_forecast)
        if not baseline_risks.empty:
            baseline_windows.append(baseline_risks.rename(columns={"start_ts":"start_ts","end_ts":"end_ts"}))
        combined["model_risk"] = combined.predicted_margin_kw.lt(0).astype(int)
        combined["model_decision"] = mark_windows(index, proposed).astype(int)
        combined["baseline_risk"] = combined.baseline_margin_kw.lt(0).astype(int)
        slots.append(combined)
        issue_times[day] = issue_timestamp(day)
    slot_results = pd.concat(slots, ignore_index=True)
    proposed_all = pd.concat(recommendations, ignore_index=True) if recommendations else pd.DataFrame(columns=["start_ts","end_ts"])
    daily_forecast_metrics = pd.DataFrame(daily_forecast_metrics)
    metric_columns = [column for column in daily_forecast_metrics if column not in {"forecast_day","target","method"}]
    forecast_results = daily_forecast_metrics.groupby(["target","method"],as_index=False)[metric_columns].mean()
    valid_peak = slot_results.evaluation_valid_peak.astype(bool)
    interval_detection = {
        "forecast_negative_margin_vs_post_dr_peak":classification_metrics(slot_results.loc[valid_peak,"actual_peak"],slot_results.loc[valid_peak,"model_risk"]),
        "recommended_events_vs_post_dr_peak":classification_metrics(slot_results.loc[valid_peak,"actual_peak"],slot_results.loc[valid_peak,"model_decision"]),
        "seasonal_margin_rule_vs_post_dr_peak":classification_metrics(slot_results.loc[valid_peak,"actual_peak"],slot_results.loc[valid_peak,"baseline_risk"]),
        "recommended_events_vs_historical_activation":classification_metrics(slot_results.actual_event,slot_results.model_decision),
    }
    evaluation_peak = slot_results.actual_peak.where(valid_peak,0)
    actual_peak_windows = boolean_windows(slot_results.timestamp, evaluation_peak)
    historical_events = load_events()
    historical_events = historical_events.loc[(historical_events.start_ts < TEST_DATES[-1]+pd.DateOffset(days=1)) &
                                                (historical_events.end_ts > TEST_DATES[0])]
    event_metrics = {
        "recommended_vs_post_dr_peak":event_matching_metrics(proposed_all, actual_peak_windows,
            replay_days=len(TEST_DATES), issue_times=issue_times),
        "recommended_vs_historical_activation":event_matching_metrics(proposed_all, historical_events,
            replay_days=len(TEST_DATES), issue_times=issue_times),
    }
    truth = load_ground_truth_for_evaluation().merge(load_events()[["event_id","start_ts"]],on="event_id")
    truth = truth.loc[truth.start_ts.between(TEST_DATES[0],TEST_DATES[-1]+pd.DateOffset(days=1),inclusive="left")]
    counterfactual = {"historical_event_household_rows":len(truth),
        "historical_event_natural_energy_kwh":float(truth.true_baseline_kwh.sum()),
        "historical_event_observed_energy_kwh":float(truth.actual_kwh.sum()),
        "historical_event_savings_kwh":float((truth.true_baseline_kwh-truth.actual_kwh).sum()),
        "usage":"offline context only; never a forecast feature or operational label"}
    summary = {"model_version":bundle.metadata["model_version"],"test_period":"2025-07-01/2025-08-31",
        "replay_days":len(TEST_DATES),"forecast_metrics":forecast_results.to_dict("records"),
        "interval_detection":interval_detection,"event_detection":event_metrics,
        "decision_policy":policy_metadata(policy),"counterfactual_evaluation_only":counterfactual,
        "excluded_peak_label_slots_due_source_gaps":int((~valid_peak).sum()),
        "label_warning":"is_dr_peak and observed demand are post-intervention; historical events are operator actions, not shortage truth."}
    slot_results.to_parquet(COMMUNITY_ARTIFACT/"test_replay_slots.parquet",index=False)
    daily_forecast_metrics.to_csv(COMMUNITY_ARTIFACT/"daily_forecast_evaluation.csv",index=False)
    proposed_all.to_csv(COMMUNITY_ARTIFACT/"test_recommendations.csv",index=False)
    forecast_results.to_csv(COMMUNITY_ARTIFACT/"forecast_evaluation.csv",index=False)
    (COMMUNITY_ARTIFACT/"decision_policy.json").write_text(json.dumps(policy_metadata(policy),indent=2),encoding="utf-8")
    (COMMUNITY_ARTIFACT/"backtest_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    metadata_path=COMMUNITY_ARTIFACT/"metadata.json"; metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["held_out_evaluation"]={"test_period":summary["test_period"],"replay_days":summary["replay_days"],
                                      "summary_file":"backtest_summary.json"}
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(forecast_results.to_string(index=False)); print(json.dumps(interval_detection,indent=2)); print(json.dumps(event_metrics,indent=2))


if __name__ == "__main__": main()
