import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from backend.intelligence.backtesting import classification_metrics, event_matching_metrics, mark_windows
from backend.intelligence.community_forecasting import (
    COMMUNITY_ARTIFACT, CommunityForecastBundle, build_causal_features, forecast_index,
    issue_timestamp, load_causal_operational_grid,
)
from backend.intelligence.decision_engine import DecisionPolicy, recommend_events
from backend.intelligence.peak_detection import boolean_windows, detect_risk_windows
from backend.services.dr_replay_service import eligible_replay_dates, load_backtest_summary, run_replay
from frontend.charts import dr_replay_chart
from frontend.pages.operator import PAGES


def synthetic_forecast(margins):
    margins=np.asarray(margins,float); supply=np.full(len(margins),10.0); demand=supply-margins
    return pd.DataFrame({"timestamp":pd.date_range("2025-07-01",periods=len(margins),freq="30min"),
        "predicted_demand_kw":demand,"predicted_supply_kw":supply,"predicted_margin_kw":margins})


def test_original_grid_is_aggregated_to_thirty_minutes_with_coverage():
    frame=load_causal_operational_grid()
    assert len(frame)==365*48
    assert frame.timestamp.diff().dropna().eq(pd.offsets.Minute(30)).all()
    assert "households_consumption_kw_coverage" in frame
    assert frame.households_consumption_kw_coverage.between(0,1).all()


def test_issuance_and_48_target_timestamps_are_aligned():
    assert issue_timestamp("2025-07-15")==pd.Timestamp("2025-07-14 14:00")
    index=forecast_index("2025-07-15")
    assert len(index)==48 and index[0]==pd.Timestamp("2025-07-15")
    assert index[-1]==pd.Timestamp("2025-07-15 23:30")


def test_operator_eligible_range_is_data_derived_and_excludes_source_gap_days():
    dates = eligible_replay_dates()
    assert dates.min() == pd.Timestamp("2025-01-09")
    assert dates.max() == pd.Timestamp("2025-12-31")
    assert pd.Timestamp("2025-02-05") not in dates


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,reason="optional community ML runtime")
@pytest.mark.parametrize("day,status", [
    ("2025-01-09", "retrospective"),
    ("2025-06-15", "retrospective"),
    ("2025-07-15", "held_out"),
    ("2025-12-31", "retrospective"),
])
def test_operator_full_year_replay_boundaries(day, status):
    result = run_replay(day)
    assert len(result["forecast"]) == 48
    assert result["forecast_status"] == status
    assert result["prediction_issue_timestamp"] == pd.Timestamp(day) - pd.DateOffset(days=1) + pd.DateOffset(hours=14)
    assert np.allclose(result["forecast"].predicted_supply_kw,
                       result["forecast"].predicted_steg_kw + result["forecast"].predicted_pv_kw)


def test_operator_replay_rejects_insufficient_and_internal_gap_dates():
    with pytest.raises(ValueError, match="sufficient causal history"):
        run_replay("2025-01-08")
    with pytest.raises(ValueError, match="coverage threshold"):
        run_replay("2025-02-05")


def test_future_injection_cannot_change_causal_features_or_prediction():
    frame=load_causal_operational_grid().copy(); day=pd.Timestamp("2025-07-15"); cutoff=issue_timestamp(day)
    before=build_causal_features(frame,day,"households_consumption_kw")
    changed=frame.copy(); changed.loc[changed.timestamp>=cutoff,"households_consumption_kw"]=1_000_000
    after=build_causal_features(changed,day,"households_consumption_kw")
    assert np.array_equal(before,after)
    if importlib.util.find_spec("sklearn") and (COMMUNITY_ARTIFACT/"models.joblib").is_file():
        model=CommunityForecastBundle().models["households_consumption_kw"]
        assert np.array_equal(model.predict(before),model.predict(after))


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,reason="optional community ML runtime")
def test_saved_bundle_loads_and_outputs_48_kw_slots():
    bundle=CommunityForecastBundle(); result=bundle.predict_day(load_causal_operational_grid(),"2025-07-15")
    assert len(result)==48
    assert result[["predicted_demand_kw","predicted_steg_kw","predicted_pv_kw"]].ge(0).all().all()
    assert np.allclose(result.predicted_supply_kw,result.predicted_steg_kw+result.predicted_pv_kw)
    assert np.allclose(result.predicted_margin_kw,result.predicted_supply_kw-result.predicted_demand_kw)
    assert bundle.metadata["targets"]["households_consumption_kw"]=="kW"


def test_operator_replay_chart_shows_combined_supply_not_components():
    timestamps = pd.date_range("2025-07-01", periods=4, freq="30min")
    frame = pd.DataFrame({
        "timestamp": timestamps, "actual_demand_kw": [8, 9, 10, 9],
        "actual_supply_kw": [10, 10, 10, 10], "predicted_demand_kw": [8, 9, 11, 9],
        "predicted_supply_kw": [10, 10, 9, 10], "predicted_steg_kw": [8, 8, 7, 8],
        "predicted_pv_kw": [2, 2, 2, 2],
    })
    figure = dr_replay_chart(frame, pd.DataFrame(columns=["start_ts", "end_ts"]))
    names = [trace.name for trace in figure.data]
    assert names == ["Actual observed demand", "Actual modeled supply",
                     "Predicted community demand", "Predicted total available supply"]
    assert figure.data[2].line.color == "#2563EB"
    assert figure.data[3].line.color == "#059669"


def test_risk_slots_are_grouped_into_consecutive_windows():
    risks=detect_risk_windows(synthetic_forecast([2,-2,-3,1,-1,-1,-1,2]))
    assert len(risks)==2
    assert risks.iloc[0].duration_minutes==60
    assert risks.iloc[0].max_shortfall_kw==3
    assert risks.iloc[1].shortage_energy_kwh==pytest.approx(1.5)


def test_decision_thresholds_max_duration_and_non_overlap():
    forecast=synthetic_forecast([-3]*10)
    risks=detect_risk_windows(forecast)
    policy=DecisionPolicy(min_deficit_kw=2,min_duration_minutes=30,min_shortage_energy_kwh=.1,
                          max_event_duration_minutes=120,min_gap_minutes=60,max_events_per_day=2)
    events=recommend_events(forecast,risks,issue_time="2025-06-30 14:00",model_version="test",policy=policy)
    assert len(events)==1 and events.iloc[0].duration_minutes==120
    strict=DecisionPolicy(min_deficit_kw=4,min_duration_minutes=30,min_shortage_energy_kwh=.1)
    assert recommend_events(forecast,risks,issue_time="2025-06-30 14:00",model_version="test",policy=strict).empty


def test_multiple_recommendations_respect_gap_and_daily_limit():
    forecast=synthetic_forecast([-3,-3,2,2,-4,-4,2,2,-5,-5])
    risks=detect_risk_windows(forecast)
    policy=DecisionPolicy(min_deficit_kw=1,min_duration_minutes=30,min_shortage_energy_kwh=.1,
                          max_event_duration_minutes=60,min_gap_minutes=30,max_events_per_day=2)
    events=recommend_events(forecast,risks,issue_time="2025-06-30 14:00",model_version="test",policy=policy)
    assert len(events)<=2
    if len(events)==2: assert events.iloc[1].start_ts>=events.iloc[0].end_ts+pd.offsets.Minute(30)


def test_classification_and_event_matching_metrics():
    metrics=classification_metrics([1,1,0,0],[1,0,1,0])
    assert metrics["true_positive"]==1 and metrics["false_positive"]==1 and metrics["false_negative"]==1
    times=pd.date_range("2025-07-01",periods=4,freq="30min")
    actual=boolean_windows(times,[0,1,1,0]); predicted=pd.DataFrame({"start_ts":[times[2]],"end_ts":[times[3]]})
    event=event_matching_metrics(predicted,actual,replay_days=1)
    assert event["event_recall"]==1 and event["mean_best_overlap_minutes"]==30
    assert mark_windows(times,predicted).sum()==1


def test_artifact_splits_are_chronological_and_scalers_are_training_only():
    metadata=json.loads((COMMUNITY_ARTIFACT/"metadata.json").read_text())
    assert metadata["split"]=={"train":"2025-01-09/2025-05-31","validation":"2025-06-01/2025-06-30","test":"2025-07-01/2025-08-31"}
    assert "training split" in metadata["normalization"]
    assert "excluded" in metadata["intervention_strategy"]


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,reason="optional community ML runtime")
def test_held_out_replay_keeps_history_and_recommendations_separate():
    result=run_replay("2025-07-02")
    assert len(result["forecast"])==48
    assert result["prediction_issue_timestamp"]==pd.Timestamp("2025-07-01 14:00")
    assert result["recommended_dr_events"] is not result["historical_operator_events"]
    assert "Simulated" in result["status"]
    assert len(result["recommended_dr_events"])==1


def test_saved_backtest_has_real_model_baseline_and_label_separation():
    summary=load_backtest_summary()
    assert summary["replay_days"]==62
    methods={(row["target"],row["method"]) for row in summary["forecast_metrics"]}
    assert ("demand","model") in methods and ("demand","baseline") in methods
    assert "post-intervention" in summary["label_warning"]
    assert summary["counterfactual_evaluation_only"]["usage"].startswith("offline")


def test_operator_navigation_adds_dr_page_without_data_explorer():
    assert list(PAGES)==["Community Overview","Community Grid","DR Events","Household Analytics","DR Detection & Forecasting"]
    assert "Dataset Explorer" not in PAGES
