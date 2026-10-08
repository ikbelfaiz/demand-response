import pandas as pd
from backend.data.aggregation import operational_grid


def test_network_loss_not_applied_again():
    raw=pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=30,freq="1min"),"households_consumption_kw":[103.]*30,"steg_production_kw":[80.]*30,"pv_production_kw":[30.]*30,"temperature_c":[20.]*30,"is_holiday":[0]*30,"is_ramadan":[0]*30,"is_dr_event":[0]*30,"is_dr_peak":[0]*30})
    out=operational_grid(raw); assert out.households_consumption_kw.iloc[0]==103.; assert out.available_supply_kw.iloc[0]==110.; assert out.margin_kw.iloc[0]==7.
