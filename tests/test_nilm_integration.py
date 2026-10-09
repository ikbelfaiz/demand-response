import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import DEFAULT_NILM_CHECKPOINT, PROJECT_ROOT
from backend.intelligence.nilm import NILMArtifactError, _validate, artifact_identity, load_nilm_runtime
from backend.services.nilm_service import _hourly, export_csv, model_status, predict_appliances
from nilm_research.data import NILMRepository, prepare_partition
from nilm_research.evaluate import predict_partition


def test_checkpoint_is_tcn_with_saved_order_units_and_transforms():
    runtime = load_nilm_runtime(DEFAULT_NILM_CHECKPOINT, "cpu")
    assert runtime.metadata["model_kind"] == "TCN"
    assert runtime.metadata["appliance_order"] == ["ac", "water_heater", "washing_machine"]
    assert runtime.metadata["units"] == "watts"
    assert runtime.metadata["context_length"] == 256
    assert runtime.transforms["fitted_split"] == "train"


def test_missing_checkpoint_is_explicit():
    missing = PROJECT_ROOT / "nilm_research" / "outputs" / "missing-test-checkpoint.pt"
    with pytest.raises(NILMArtifactError, match="not found"):
        load_nilm_runtime(missing, "cpu")
    available, reason = model_status(checkpoint=missing, device="cpu")
    assert not available and "not found" in reason


def test_non_tcn_model_is_rejected():
    runtime = load_nilm_runtime(DEFAULT_NILM_CHECKPOINT, "cpu")
    payload = {"model_kind": "cnn_baseline", "transforms": dict(runtime.transforms),
               "config": {key: __import__("copy").deepcopy(value) for key, value in runtime.config.items()}}
    with pytest.raises(NILMArtifactError, match="not a causal TCN"):
        _validate(payload, runtime.model)


def test_backend_matches_shared_standalone_prediction():
    result = predict_appliances("C005", "2025-07-15 12:00", "2025-07-15 12:10", device="cpu")
    runtime = load_nilm_runtime(DEFAULT_NILM_CHECKPOINT, "cpu")
    cfg = {key: __import__("copy").deepcopy(value) for key, value in runtime.config.items()}
    cfg["splits"]["check"] = ["2025-07-15 07:45", "2025-07-15 12:10"]
    part = prepare_partition(NILMRepository(cfg), cfg, dict(runtime.transforms), "C005", "check")
    power, _, valid = predict_partition(runtime.model, part, 256, 8192, runtime.device)
    selected = (part.timestamps >= np.datetime64("2025-07-15T12:00"))
    np.testing.assert_allclose(result.minute.predicted_ac_power_w, power[selected, 0], rtol=0, atol=1e-5)
    assert result.minute.quality_status.eq("ok").to_numpy().tolist() == valid[selected].tolist()


def test_interval_unlabeled_home_and_null_references():
    result = predict_appliances("C001", "2025-07-15 12:00", "2025-07-15 12:03", device="cpu")
    assert result.minute.timestamp.tolist() == list(pd.date_range("2025-07-15 12:00", periods=3, freq="min"))
    assert result.minute.reference_ac_power_w.isna().all()
    assert result.minute.predicted_ac_power_w.notna().any()


def test_invalid_context_has_null_predictions_not_fake_zero():
    result = predict_appliances("C005", "2025-01-01 00:00", "2025-01-01 00:10", device="cpu")
    assert result.minute.quality_reason.eq("insufficient_historical_context").all()
    assert result.minute.predicted_ac_power_w.isna().all()
    assert result.summary["appliances"]["ac"]["observed_estimated_energy_kwh"] is None


def test_partial_hour_energy_and_json_contract():
    minute = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 00:15", periods=30, freq="min"),
        "quality_status": ["ok"] * 15 + ["unavailable"] * 15,
        **{f"predicted_{name}_power_w": [1000.0] * 15 + [np.nan] * 15
           for name in ("ac", "water_heater", "washing_machine")},
    })
    row = _hourly(minute).iloc[0]
    assert row.ac_energy_kwh == pytest.approx(0.25)
    assert row.coverage_fraction == 0.5
    assert row.full_hour_coverage_fraction == 0.25
    result = predict_appliances("C005", "2025-01-01 00:00", "2025-01-01 00:02", device="cpu")
    encoded = json.dumps(result.json_safe())
    assert "NaN" not in encoded and "+01:00" in encoded
    assert export_csv(result, "minute").startswith(b"timestamp,client_id")


def test_cache_identity_changes_when_artifact_changes(tmp_path):
    path = tmp_path / "identity.pt"
    path.write_bytes(b"one")
    first = artifact_identity(path)
    path.write_bytes(b"longer")
    assert artifact_identity(path) != first
