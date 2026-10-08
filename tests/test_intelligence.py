import pandas as pd
import pytest
from backend.intelligence.baseline import estimate_baseline
from backend.intelligence.forecasting import build_history_at_issuance, empty_forecast
from backend.intelligence.model_registry import MODEL_REGISTRY
from backend.intelligence.nilm import estimate_appliances
from backend.intelligence.orchestrator import status


def test_registry_status_matches_artifact_availability():
    assert ("energy_ttm" in MODEL_REGISTRY.status()) == MODEL_REGISTRY.available()
    assert ("community_dr" in MODEL_REGISTRY.status()) == MODEL_REGISTRY.community_available()
    assert empty_forecast().empty


def test_future_observations_excluded():
    frame=pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=4,freq="30min")})
    result=build_history_at_issuance(frame,"2025-01-01 01:00"); assert result.timestamp.max()<pd.Timestamp("2025-01-01 01:00")


def test_untrained_outputs_are_not_fabricated():
    with pytest.raises(NotImplementedError): estimate_baseline()
    with pytest.raises(NotImplementedError): estimate_appliances()
