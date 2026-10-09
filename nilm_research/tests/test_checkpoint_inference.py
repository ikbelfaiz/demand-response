import csv
import numpy as np
import torch

from nilm_research.checkpoint import load_checkpoint, save_checkpoint
from nilm_research.inference import hourly_energy_rows, run_inference
from nilm_research.model import build_model


def test_energy_conversion_and_partial_coverage():
    ts = np.arange(np.datetime64("2025-01-01T00:00"), np.datetime64("2025-01-01T01:00"), np.timedelta64(1, "m"))
    power = np.full((60, 3), 1000.0, dtype=np.float32)
    valid = np.ones(60, dtype=bool); valid[30:] = False
    row = hourly_energy_rows(ts, power, valid)[0]
    assert row["coverage_fraction"] == 0.5
    assert row["ac_energy_kwh"] == 0.5


def test_checkpoint_reload_prediction_consistency(prepared, cfg, tmp_path):
    *_, transforms, _, part = prepared
    model = build_model("cnn_baseline", transforms, cfg["model"]).eval()
    x = torch.from_numpy(part.inputs[:, :256]).unsqueeze(0)
    with torch.no_grad(): before = model(x)["power_w"]
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, 0, 1.0, cfg, transforms, "cnn_baseline", [])
    loaded, _ = load_checkpoint(path)
    with torch.no_grad(): after = loaded(x)["power_w"]
    torch.testing.assert_close(before, after)


def test_standalone_inference_reports_insufficient_history(prepared, cfg, tmp_path):
    _, panel, _, transforms, _, _ = prepared
    model = build_model("cnn_baseline", transforms, cfg["model"])
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, None, 0, 1.0, cfg, transforms, "cnn_baseline", [])
    output = tmp_path / "predictions.csv"
    status = run_inference(checkpoint, panel[0], "2025-01-01 00:00:00", "2025-01-01 02:00:00", output, "cpu")
    assert status["rows"] == 120
    assert status["valid_predictions"] == 0
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(r["quality_status"] == "insufficient_historical_context" for r in rows)
