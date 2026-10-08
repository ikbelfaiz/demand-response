import pandas as pd
import pytest

from backend.data.preprocessing import prepare_dataset
from backend.services.analytics_service import chart_series, filter_dates
from backend.services.energy_service import daily_consumption, energy_summary, integrate_power


def test_energy_integration_at_one_minute(config, raw):
    data = prepare_dataset(raw, config)
    assert integrate_power(data, "household_power_kw") == pytest.approx(4 / 60)


def test_energy_integration_uses_elapsed_time():
    data = pd.DataFrame({"timestamp":pd.to_datetime(["2025-01-01 00:00","2025-01-01 00:30","2025-01-01 01:30"]),
                         "power":[2.0,2.0,2.0]})
    # Forward durations are 0.5 h and 1 h; the final sample uses their 0.75 h median.
    assert integrate_power(data, "power") == pytest.approx(4.5)


def test_daily_and_peak_calculations(config, raw):
    data = prepare_dataset(raw, config)
    summary = energy_summary(data)
    assert summary["peak_kw"] == 2.0
    assert daily_consumption(data).energy_kwh.iloc[0] == pytest.approx(4 / 60)


def test_empty_and_invalid_date_filter(config, raw):
    data = prepare_dataset(raw, config)
    assert filter_dates(data, "2030-01-01").empty
    with pytest.raises(Exception):
        filter_dates(data, "invalid")


def test_chart_resampling_does_not_mutate_analytics(config, raw):
    data = prepare_dataset(raw, config)
    before = integrate_power(data, "household_power_kw")
    chart = chart_series(data, "2min")
    assert len(chart) == 2
    assert integrate_power(data, "household_power_kw") == before
    assert chart.household_power_kw.iloc[0] == pytest.approx(1.5)
