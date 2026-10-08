import pandas as pd

from backend.config import get_v3_paths
from backend.data import assert_v3_files, load_events, load_household_info, load_participation, resample_power
from backend.data.validator import validate_v3


def test_v3_file_discovery():
    assert_v3_files(); assert all(p.is_file() for p in get_v3_paths().inventory().values())


def test_actual_households_and_relationships():
    info, events, part=load_household_info(),load_events(),load_participation()
    assert info.client_id.tolist()==[f"C{i:03d}" for i in range(1,51)]
    assert len(events)==36 and len(part)==36*50
    assert set(part.event_id)==set(events.event_id); assert set(part.client_id)==set(info.client_id)
    assert info.has_submeter.sum()==10


def test_half_hour_power_and_energy_units(minute_power):
    out=resample_power(minute_power,["power_kw"])
    assert out.power_kw.tolist()==[2.0,2.0]
    assert out.power_energy_kwh.tolist()==[1.0,1.0]


def test_valid_zero_is_measurement_not_missing():
    frame=pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=30,freq="1min"),"power_kw":[0.0]*30})
    out=resample_power(frame,["power_kw"])
    assert out.loc[0,"power_kw"]==0.0; assert out.loc[0,"power_energy_kwh"]==0.0


def test_dataset_validation_contract():
    report=validate_v3(); assert report["household_count"]==50; assert report["event_count"]==36
    assert report["grid_complete_timeline"]; assert all(v==0 for v in report["relationships"].values())
