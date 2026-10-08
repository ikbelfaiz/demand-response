import pandas as pd

from frontend import charts
from frontend.styles import BORDERS, PLOTLY_THEME, TEXT


def sample_frame():
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="1h"),
        "household_power_kw": [0.2, 0.5, 0.3],
        "zone_demand_mw": [500.0, 520.0, 510.0],
        "system_production_mw": [900.0, 905.0, 902.0],
        "zone_pv_production_mw": [0.0, 5.0, 10.0],
    })


def assert_readable_light_theme(figure):
    assert figure.layout.paper_bgcolor == "#FFFFFF"
    assert figure.layout.plot_bgcolor == "#FFFFFF"
    assert figure.layout.font.color == TEXT["secondary"]
    assert figure.layout.xaxis.tickfont.color == TEXT["secondary"]
    assert figure.layout.yaxis.tickfont.color == TEXT["secondary"]
    assert figure.layout.xaxis.title.font.color == TEXT["primary"]
    assert figure.layout.yaxis.title.font.color == TEXT["primary"]
    assert figure.layout.legend.font.color == TEXT["primary"]
    assert figure.layout.annotationdefaults.font.color == TEXT["primary"]
    assert figure.layout.hoverlabel.bgcolor == TEXT["primary"]
    assert figure.layout.hoverlabel.font.color == TEXT["on_dark"]
    assert figure.layout.xaxis.gridcolor == BORDERS["grid"]
    assert figure.layout.yaxis.gridcolor == BORDERS["grid"]


def test_all_standard_chart_types_inherit_readable_theme():
    data = sample_frame()
    profile = pd.DataFrame({"hour":[0,1],"average_power_kw":[.2,.3],"p95_power_kw":[.4,.5]})
    daily = pd.DataFrame({"date":pd.to_datetime(["2025-01-01"]),"energy_kwh":[4.2],"observed_coverage":[.99]})
    appliances = data.assign(ac_power_kw=[0,.2,.1])
    breakdown = pd.DataFrame({"appliance":["AC"],"energy_kwh":[.3],"coverage":[1.0]})
    events = pd.DataFrame({"start":pd.to_datetime(["2025-01-01"]),"duration_minutes":[120],"peak_zone_mw":[520.0],"peak_household_kw":[.5]})
    figures = [
        charts.household_load(data), charts.regional_demand(data), charts.production_demand(data),
        charts.pv_production(data), charts.daily_energy(daily), charts.hourly_pattern(profile),
        charts.appliance_stack(appliances,{"ac_power_kw":"Air conditioning"}),
        charts.appliance_energy(breakdown), charts.demand_distribution(data.household_power_kw),
        charts.historical_labels(events),
    ]
    for figure in figures:
        assert_readable_light_theme(figure)


def test_heatmap_colorbar_text_is_explicitly_dark():
    matrix = pd.DataFrame([[.2,.3]], index=["2025-01-01"], columns=[0,1])
    figure = charts.load_heatmap(matrix)
    assert_readable_light_theme(figure)
    colorbar = figure.data[0].colorbar
    assert colorbar.title.font.color == PLOTLY_THEME["primary_text"]
    assert colorbar.tickfont.color == PLOTLY_THEME["secondary_text"]
