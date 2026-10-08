import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("page", ["Community Overview", "Community Grid", "DR Events", "Household Analytics"])
def test_operator_pages_render(page):
    app=AppTest.from_file(APP,default_timeout=120).run()
    next(widget for widget in app.radio if widget.label=="Page").set_value(page); app.run(timeout=120)
    assert not app.exception


@pytest.mark.parametrize("page", ["My Energy", "My Household", "My Appliances", "My DR Participation", "Preferences"])
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
