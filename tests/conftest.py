import pandas as pd
import pytest


@pytest.fixture
def minute_power():
    return pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=60,freq="1min"),"power_kw":[2.0]*60})
