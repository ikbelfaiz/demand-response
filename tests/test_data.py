import pandas as pd
import pytest

from backend.data.loader import load_dataset
from backend.data.preprocessing import prepare_dataset
from backend.data.validator import validate_dataset


def test_dataset_loading(config, raw):
    raw.to_csv(config.path, index=False)
    loaded = load_dataset(config)
    assert len(loaded) == 4
    assert loaded.columns.tolist() == raw.columns.tolist()


def test_timestamp_parsing_and_units(config, raw):
    clean = prepare_dataset(raw, config)
    assert pd.api.types.is_datetime64_any_dtype(clean.timestamp)
    assert clean.household_power_kw.tolist() == [1.0, 2.0, 1.0, 0.0]
    assert clean.zone_demand_mw.iloc[0] == 500.0


def test_missing_column_rejected(config, raw):
    with pytest.raises(ValueError, match="consumption"):
        validate_dataset(raw.drop(columns="load_w"), config)


def test_gaps_and_duplicates_reported(config, raw):
    altered = pd.concat([raw.iloc[[0]], raw.iloc[[2]], raw.iloc[[2]], raw.iloc[[3]]], ignore_index=True)
    report = validate_dataset(altered, config)
    assert report.duplicate_timestamps == 1
    assert report.missing_intervals == 1


def test_missing_sensor_interpolated_with_provenance(config, raw):
    clean = prepare_dataset(raw, config)
    assert clean.ac_power_kw.iloc[2] == pytest.approx(.25)
    assert bool(clean.ac_power_was_missing.iloc[2])


def test_long_sensor_outage_is_not_fabricated(config):
    minutes = 185
    raw = pd.DataFrame({"time":pd.date_range("2025-01-01", periods=minutes, freq="1min").astype(str),
                        "load_w":[1000.0] + [None] * 183 + [1000.0], "event":0})
    clean = prepare_dataset(raw, config)
    assert clean.household_power_kw.isna().sum() == 183


def test_invalid_timestamp(config, raw):
    raw.loc[1, "time"] = "not a date"
    with pytest.raises(ValueError, match="timestamps"):
        validate_dataset(raw, config)

