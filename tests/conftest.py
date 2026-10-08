import pandas as pd
import pytest

from backend.config import DatasetConfig


@pytest.fixture
def config(tmp_path):
    return DatasetConfig(
        name="test", path=tmp_path / "test.csv",
        column_map={"timestamp":"time", "household_power":"load_w", "ac_power":"ac_w",
                    "dr_event":"event", "zone_demand":"zone_mw"},
        units={"household_power":"W", "ac_power":"W", "zone_demand":"MW"},
    )


@pytest.fixture
def raw():
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=4, freq="1min").astype(str),
        "load_w": [1000.0, 2000.0, 1000.0, 0.0], "ac_w":[0.0, 500.0, None, 0.0],
        "event":[0,1,1,0], "zone_mw":[500.0,501.0,502.0,503.0],
    })

