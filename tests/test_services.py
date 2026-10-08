import pandas as pd
import pytest

from backend.data.preprocessing import prepare_dataset
from backend.services.data_quality_service import descriptive_statistics, missing_statistics, sensor_outages
from backend.services.energy_service import energy_summary
from backend.services.historical_event_service import historical_event_labels
from backend.services.household_service import appliance_breakdown, appliance_columns, hourly_profile
from backend.services.regional_service import regional_summary, reported_supply_comparison


def test_energy_summary_reports_incomplete_coverage(config, raw):
    raw.loc[1, "load_w"] = None
    # Make the gap too long for interpolation by building a longer frame.
    long = pd.DataFrame({"time":pd.date_range("2025-01-01",periods=185,freq="1min").astype(str),
                         "load_w":[1000.0]+[None]*183+[1000.0],"event":0,"zone_mw":500.0})
    data=prepare_dataset(long,config); summary=energy_summary(data)
    assert summary["coverage"] < 1
    assert not summary["energy_complete"]
    assert summary["valid_samples"] == 2


def test_appliance_discovery_and_breakdown(config, raw):
    data=prepare_dataset(raw,config)
    assert appliance_columns(data)==["ac_power_kw"]
    result=appliance_breakdown(data)
    assert result.appliance.tolist()==["Air conditioning"]
    assert result.energy_kwh.iloc[0] > 0


def test_hourly_household_profile(config, raw):
    result=hourly_profile(prepare_dataset(raw,config))
    assert result.hour.tolist()==[0]
    assert result.average_power_kw.iloc[0]==pytest.approx(1.0)


def test_regional_measurements_remain_mw(config, raw):
    data=prepare_dataset(raw,config); result=regional_summary(data)
    assert result["zone_demand_mw_average"]==pytest.approx(501.5)
    assert result["zone_demand_mw_peak"]==503.0


def test_reported_supply_comparison_requires_all_signals(config, raw):
    data=prepare_dataset(raw,config)
    result=reported_supply_comparison(data)
    assert "reported_production_minus_demand_mw" not in result


def test_quality_statistics_preserve_original_missing(config, raw):
    data=prepare_dataset(raw,config)
    stats=missing_statistics(data).set_index("channel")
    assert stats.loc["ac_power_kw","original_missing"]==1
    assert stats.loc["ac_power_kw","remaining_missing"]==0
    outages=sensor_outages(data)
    assert outages.iloc[0].channel=="ac_power_kw"
    assert not descriptive_statistics(data).empty


def test_historical_labels_are_retrospective_data(config, raw):
    history=historical_event_labels(prepare_dataset(raw,config))
    assert len(history)==1
    assert set(history.columns)=={"start","end","duration_minutes","peak_household_kw","peak_zone_mw"}
    assert "prediction" not in " ".join(history.columns)

