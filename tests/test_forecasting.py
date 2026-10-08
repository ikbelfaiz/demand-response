import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.intelligence.forecast_features import (
    CONTEXT_SLOTS, HORIZON_SLOTS, build_window, chronological_split,
    normalize_context, seasonal_persistence,
)
from backend.intelligence.forecast_metrics import forecast_kpis, forecast_metrics
from backend.intelligence.consumer_forecast_data import build_consumer_window
from backend.intelligence.model_registry import ModelArtifactError, ModelRegistry
from backend.intelligence.model_registry import DEFAULT_FORECAST_ARTIFACT
from backend.services import forecast_service
from frontend.charts import forecast_chart
from frontend.pages.consumer import PAGES as CONSUMER_PAGES
from frontend.pages.operator import PAGES as OPERATOR_PAGES


def half_hour_frame(days=8, start="2025-01-01", value=None):
    timestamps = pd.date_range(start, periods=days * 48, freq="30min")
    values = np.arange(len(timestamps), dtype=np.float32) / 100 if value is None else np.full(len(timestamps), value)
    return pd.DataFrame({"timestamp": timestamps, "aggregate_power_kw": values})


def test_exact_window_shapes_alignment_and_no_future_leakage():
    frame = half_hour_frame()
    window = build_window(frame, "C001", "2025-01-08", require_target=True)
    assert window.context_kw.shape == (CONTEXT_SLOTS,)
    assert window.target_kw.shape == (HORIZON_SLOTS,)
    assert window.context_timestamps[-1] < window.forecast_date
    assert window.target_timestamps[0] == pd.Timestamp("2025-01-08")
    assert np.array_equal(window.context_kw, frame.aggregate_power_kw.iloc[:336])


def test_missing_context_is_rejected_not_imputed():
    frame = half_hour_frame().drop(index=120)
    with pytest.raises(ValueError, match="incomplete"):
        build_window(frame, "C001", "2025-01-08")


def test_context_scaling_uses_history_only():
    frame = half_hour_frame()
    window = build_window(frame, "C001", "2025-01-08", require_target=True)
    scaled, mean, std = normalize_context(window.context_kw)
    assert scaled.shape == (336,)
    assert mean == pytest.approx(float(frame.aggregate_power_kw.iloc[:336].mean()))
    assert mean != pytest.approx(float(frame.aggregate_power_kw.mean()))
    assert std > 0


def test_household_specific_scaling_is_independent_of_absolute_load_level():
    pattern = np.sin(np.arange(336) * np.pi / 24).astype(np.float32)
    low, _, _ = normalize_context(pattern + 1.0)
    high, _, _ = normalize_context(pattern * 4.0 + 12.0)
    assert np.allclose(low, high, atol=2e-6)


def test_consumer_window_context_does_not_use_future_target_values():
    timestamps = pd.date_range("2025-01-01", periods=8 * 48, freq="30min")
    base = pd.DataFrame({"timestamp": timestamps, "context_power_kw": 1.0,
                         "aggregate_power_kw": 1.0, "source_coverage": 1.0})
    changed = base.copy()
    changed.loc[changed.timestamp >= pd.Timestamp("2025-01-08"), "aggregate_power_kw"] = 999.0
    first = build_consumer_window(base, "C001", "2025-01-08", require_target=True)
    second = build_consumer_window(changed, "C001", "2025-01-08", require_target=True)
    assert np.array_equal(first.context_kw, second.context_kw)
    assert not np.array_equal(first.target_kw, second.target_kw)


def test_chronological_split_has_disjoint_target_days():
    dates = pd.date_range("2025-01-08", "2025-12-31", freq="D")
    split = chronological_split(dates, "2025-09-30", "2025-10-31")
    assert split["train"].max() < split["validation"].min() < split["test"].min()
    assert not set(split["train"]) & set(split["test"])


def test_seasonal_persistence_is_same_slots_seven_days_earlier():
    context = np.arange(336, dtype=np.float32)
    assert np.array_equal(seasonal_persistence(context), np.arange(48, dtype=np.float32))


def test_metrics_energy_peak_and_zero_actual_handling():
    actual = np.zeros(48); predicted = np.ones(48)
    metrics = forecast_metrics(actual, predicted)
    assert metrics["mae_kw"] == 1
    assert metrics["rmse_kw"] == 1
    assert np.isnan(metrics["wape_pct"])
    assert metrics["daily_energy_error_kwh"] == 24
    assert metrics["daily_energy_absolute_error_kwh"] == 24
    kpis = forecast_kpis(predicted)
    assert kpis["daily_energy_kwh"] == 24
    assert kpis["peak_power_kw"] == 1


def test_peak_timing_error_is_absolute_and_bias_preserves_direction():
    actual = np.array([0.0, 0.0, 2.0, 0.0])
    predicted = np.array([0.0, 3.0, 0.0, 0.0])
    metrics = forecast_metrics(actual, predicted)
    assert metrics["peak_timing_error_minutes"] == 30.0
    assert metrics["peak_timing_bias_minutes"] == -30.0
    assert metrics["peak_magnitude_error_kw"] == 1.0
    assert metrics["peak_magnitude_bias_kw"] == 1.0


def test_registry_rejects_missing_and_incompatible_artifacts(tmp_path):
    with pytest.raises(ModelArtifactError, match="train_forecasting"):
        ModelRegistry(tmp_path).metadata()
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    metadata = {"model_name":"x","model_version":"1","context_length":168,
                "prediction_length":24,"frequency":"1h","target":"x",
                "model_directory":"model","test_start":"2025-11-01"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ModelArtifactError, match="336-to-48"):
        ModelRegistry(tmp_path).metadata()


class MockPredictor:
    metadata = {"model_name":"Energy-TTM","model_version":"test","test_start":"2025-11-01",
                "split":{"train":"2025-01-08/2025-09-30",
                         "validation":"2025-10-01/2025-10-31",
                         "test":"2025-11-01/2025-12-31"}}
    def predict(self, context_kw):
        assert len(context_kw) == 336
        return np.full(48, 2.0, dtype=np.float32)


def minute_source(start, days):
    ts = pd.date_range(start, periods=days * 1440, freq="1min")
    return pd.DataFrame({"timestamp":ts,"client_id":"C007","aggregate_power_w":2000.0,
                         "ac_power_w":0.0,"water_heater_power_w":0.0,"washing_machine_power_w":0.0})


def test_forecast_service_household_units_and_historical_comparison(monkeypatch):
    source = minute_source("2025-12-24", 8)
    monkeypatch.setattr(forecast_service, "load_household", lambda *_a, **_k: source)
    result = forecast_service.forecast_next_day("C007", "2025-12-31", predictor=MockPredictor())
    assert result["client_id"] == "C007"
    assert result["actual_comparison"] == "held_out"
    assert result["forecast_status_label"] == "Historical replay — out-of-sample evaluation."
    assert len(result["series"]) == 48
    assert result["series"].predicted_energy_kwh.eq(1.0).all()
    assert result["kpis"]["daily_energy_kwh"] == 48
    assert not any("persistence" in column for column in result["series"].columns)


def test_consumer_january_1_is_rejected_for_missing_context():
    with pytest.raises(ValueError, match="incomplete"):
        forecast_service.forecast_next_day("C001", "2025-01-01", predictor=MockPredictor())


@pytest.mark.parametrize("day,expected_status", [
    ("2025-01-08", "retrospective"),
    ("2025-06-15", "retrospective"),
    ("2025-12-31", "held_out"),
])
def test_consumer_full_year_boundaries_return_48_slots(day, expected_status):
    result = forecast_service.forecast_next_day("C001", day, predictor=MockPredictor())
    assert len(result["series"]) == 48
    assert result["forecast_status"] == expected_status
    assert result["historical_input_window"]["slots"] == 336
    assert result["historical_input_window"]["end"] < result["forecast_date"]


def test_consumer_date_bounds_cover_all_50_households():
    for number in range(1, 51):
        dates = forecast_service.consumer_forecast_dates(f"C{number:03d}")
        assert dates.min() == pd.Timestamp("2025-01-08")
        assert dates.max() == pd.Timestamp("2025-12-31")


def test_missing_split_provenance_defaults_to_retrospective():
    status, label = forecast_service.forecast_classification({}, "2025-12-31")
    assert status == "retrospective"
    assert label.startswith("Retrospective prediction")


def test_forecast_service_next_day_has_no_manufactured_actual(monkeypatch):
    source = minute_source("2025-12-25", 7)
    monkeypatch.setattr(forecast_service, "load_household", lambda *_a, **_k: source)
    result = forecast_service.forecast_next_day("C007", "2026-01-01", predictor=MockPredictor())
    assert result["actual_comparison"] == "unavailable"
    assert "actual_power_kw" not in result["series"]
    assert result["metrics"] is None


def test_forecast_chart_has_separate_actual_and_dashed_prediction():
    frame = pd.DataFrame({"timestamp":pd.date_range("2025-12-01", periods=48, freq="30min"),
                          "actual_power_kw":1.0,"predicted_power_kw":2.0})
    figure = forecast_chart(frame)
    assert len(figure.data) == 2
    assert figure.data[0].connectgaps is False
    assert figure.data[1].line.dash == "dash"


def test_forecast_chart_ignores_obsolete_persistence_columns():
    frame = pd.DataFrame({"timestamp":pd.date_range("2025-12-01", periods=48, freq="30min"),
                          "actual_power_kw":1.0,"predicted_power_kw":2.0,
                          "persistence_power_kw":1.5})
    figure = forecast_chart(frame)
    assert [trace.name for trace in figure.data] == ["Actual power", "Energy-TTM forecast"]


def test_consumer_evaluation_reports_each_household_separately():
    path = Path("outputs/consumer_forecast_improvement/test_stage2_per_household.csv")
    if not path.is_file():
        pytest.skip("consumer candidate evaluation has not been run")
    frame = pd.read_csv(path)
    assert frame.client_id.nunique() == 50


def test_data_explorer_is_completely_removed_from_navigation():
    assert list(OPERATOR_PAGES) == ["Community Overview", "Community Grid", "DR Events",
                                    "Household Analytics", "DR Detection & Forecasting"]
    assert "Dataset Explorer" not in OPERATOR_PAGES


def test_consumer_navigation_has_no_preferences_page():
    assert list(CONSUMER_PAGES) == ["My Energy", "My Household", "My Appliances", "My DR Participation"]
    assert "Preferences" not in CONSUMER_PAGES


@pytest.mark.skipif(importlib.util.find_spec("tsfm_public") is None, reason="optional Energy-TTM runtime")
def test_saved_model_predictions_are_consistent_after_reload():
    if not ModelRegistry().available():
        pytest.skip("trained artifact not present")
    from backend.intelligence.forecasting import EnergyTTMForecaster
    context = np.sin(np.arange(336) * np.pi / 24).astype(np.float32) + 2
    first = EnergyTTMForecaster(DEFAULT_FORECAST_ARTIFACT).predict(context)
    second = EnergyTTMForecaster(DEFAULT_FORECAST_ARTIFACT).predict(context)
    assert first.shape == (48,)
    assert np.array_equal(first, second)
