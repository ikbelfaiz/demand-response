import pytest
import importlib.util
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("page", ["Community Overview", "Community Grid", "DR Events", "Household Analytics", "DR Detection & Forecasting"])
def test_operator_pages_render(page):
    app=AppTest.from_file(APP,default_timeout=120).run()
    next(widget for widget in app.radio if widget.label=="Page").set_value(page); app.run(timeout=120)
    assert not app.exception


@pytest.mark.parametrize("page", ["My Energy", "My Household", "My Appliances", "My DR Participation"])
def test_consumer_pages_render(page):
    app=AppTest.from_file(APP,default_timeout=120).run()
    next(widget for widget in app.radio if widget.label=="Perspective").set_value("Consumer Dashboard"); app.run(timeout=120)
    next(widget for widget in app.radio if widget.label=="Page").set_value(page); app.run(timeout=120)
    assert not app.exception


def test_my_energy_forecast_section_renders_with_generate_control():
    app=AppTest.from_file(APP,default_timeout=120).run()
    next(widget for widget in app.radio if widget.label=="Perspective").set_value("Consumer Dashboard"); app.run(timeout=120)
    assert any(item.value == "Tomorrow's Energy Forecast" for item in app.subheader)
    assert any(button.label == "Generate forecast" for button in app.button)
    assert not app.exception


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="optional community ML runtime")
def test_operator_dr_replay_executes_saved_model():
    app=AppTest.from_file(APP,default_timeout=120).run()
    next(widget for widget in app.radio if widget.label=="Page").set_value("DR Detection & Forecasting")
    app.run(timeout=120)
    next(button for button in app.button if button.label=="Run Forecast / Replay").click()
    app.run(timeout=120)
    assert not app.exception
    assert any("Simulated decision generated" in item.value for item in app.success)
    assert all("Held-out" not in item.value for item in app.subheader)
    assert all("model performance" not in item.value.lower() for item in app.subheader)
    assert all(metric.label not in {"Demand MAE", "Demand RMSE", "Supply MAE", "Supply RMSE",
                                    "Interval precision", "Interval recall", "F1"}
               for metric in app.metric)
