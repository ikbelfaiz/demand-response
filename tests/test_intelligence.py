import pandas as pd
import pytest

from backend.intelligence.feature_engineering import build_monitoring_features
from backend.intelligence.model_registry import ModelRegistry, ModelUnavailableError


def test_dashboard_model_registry_starts_empty():
    registry=ModelRegistry()
    assert registry.available()==[]
    with pytest.raises(ModelUnavailableError):
        registry.load("dr-detector","v1")


def test_registry_tracks_explicit_versions_only():
    registry=ModelRegistry(); model=object()
    registry.register("peak-forecast","2026.1",model)
    assert registry.available()==[("peak-forecast","2026.1")]
    assert registry.load("peak-forecast","2026.1") is model


def test_feature_dataset_excludes_historical_event_labels():
    frame=pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=2,freq="1h"),
                        "zone_demand_mw":[500,510],"system_production_mw":[900,905],
                        "zone_pv_production_mw":[0,0],"temperature_c":[12,12],"dr_event":[1,0]})
    features=build_monitoring_features(frame)
    assert "dr_event" not in features.frame
    assert features.label_column is None
    assert features.feature_version=="v1"
