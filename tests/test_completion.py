import json
from pathlib import Path

import pandas as pd
import pytest

from backend.config import DATA_DIR, get_v3_paths
from backend.data import load_grid, load_household, load_household_info, operational_grid, operational_household
from frontend.charts import lines


BACKUP=DATA_DIR/"original_v3_backup"


def test_backup_checksums_and_completion_record_exist():
    checks=json.loads((BACKUP/"backup_checksums.json").read_text(encoding="utf-8"))
    manifest=json.loads((BACKUP/"completion_manifest.json").read_text(encoding="utf-8"))
    assert checks==manifest["backup_sha256"]
    assert set(checks)=={"grid_1min.csv","households_1min_C001_C025.parquet","households_1min_C026_C050.parquet"}


def test_grid_is_complete_unique_and_physically_valid():
    grid=load_grid(); sensors=["temperature_c","households_consumption_kw","steg_production_kw","pv_production_kw"]
    assert len(grid)==525_600 and not grid.timestamp.duplicated().any()
    assert not grid[sensors+["is_dr_peak"]].isna().any().any()
    assert (grid[["households_consumption_kw","steg_production_kw","pv_production_kw"]]>=0).all().all()
    assert grid.temperature_c.between(-20,60).all()
    assert not any(c.endswith(("_imputed","_was_imputed")) for c in grid)


def test_existing_grid_labels_unchanged_and_missing_peaks_reconstructed():
    original=pd.read_csv(BACKUP/"grid_1min.csv")
    completed=pd.read_csv(get_v3_paths().grid)
    for column in ["is_holiday","is_ramadan","is_dr_event"]: pd.testing.assert_series_equal(original[column],completed[column],check_dtype=False)
    known=original.is_dr_peak.notna(); pd.testing.assert_series_equal(original.loc[known,"is_dr_peak"],completed.loc[known,"is_dr_peak"].astype(float),check_dtype=False)
    assert original.is_dr_peak.isna().sum()==528 and completed.is_dr_peak.isna().sum()==0


@pytest.mark.parametrize("client_id",[f"C{i:03d}" for i in range(1,51)])
def test_every_household_is_complete_and_unique(client_id):
    info=load_household_info().set_index("client_id").loc[client_id]; frame=load_household(client_id)
    assert len(frame)==525_600 and not frame.timestamp.duplicated().any()
    assert frame.timestamp.iloc[0]==pd.Timestamp("2025-01-01") and frame.timestamp.iloc[-1]==pd.Timestamp("2025-12-31 23:59")
    assert frame.aggregate_power_w.notna().all() and frame.aggregate_power_w.ge(0).all()
    for channel in ["ac_power_w","water_heater_power_w","washing_machine_power_w"]:
        if info.has_submeter: assert frame[channel].notna().all() and frame[channel].ge(0).all()
        else: assert frame[channel].isna().all()
    assert not any(c.endswith(("_imputed","_was_imputed")) for c in frame)


@pytest.mark.parametrize("client_id",["C001","C005"])
def test_december_31_operational_trace_is_complete(client_id):
    frame=operational_household(load_household(client_id,"2025-12-31","2026-01-01"))
    expected=pd.date_range("2025-12-31",periods=48,freq="30min")
    assert len(frame)==48 and frame.timestamp.tolist()==expected.tolist()
    assert frame.aggregate_power_kw.notna().all()
    if client_id=="C005": assert frame[["ac_power_kw","water_heater_power_kw","washing_machine_power_kw"]].notna().all().all()
    fig=lines(frame,[("aggregate_power_kw","Aggregate","#2563EB")],"Power (kW)")
    assert len(fig.data)==1 and len(fig.layout.shapes)==0 and not pd.isna(fig.data[0].y).any()
    if client_id=="C005":
        appliance_fig=lines(frame,[("aggregate_power_kw","Aggregate","#2563EB"),("ac_power_kw","AC","#EF4444"),
            ("water_heater_power_kw","Water heater","#06B6D4"),("washing_machine_power_kw","Washing machine","#8B5CF6")],"Power (kW)")
        assert len(appliance_fig.data)==4 and len(appliance_fig.layout.shapes)==0
        assert all(not pd.isna(trace.y).any() and trace.mode in (None,"lines") for trace in appliance_fig.data)


def test_all_grid_half_hours_are_populated():
    frame=operational_grid(load_grid())
    assert len(frame)==17_520
    assert not frame[["households_consumption_kw","steg_production_kw","pv_production_kw","temperature_c"]].isna().any().any()
