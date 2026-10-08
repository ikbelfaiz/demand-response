import pandas as pd
import pytest
from backend.services.event_service import event_table
from backend.services.household_service import summary


def test_participation_join_counts():
    events=pd.DataFrame({"event_id":[1],"start_ts":pd.to_datetime(["2025-01-01 18:00"]),"end_ts":pd.to_datetime(["2025-01-01 20:00"]),"reduction_target_pct":[20],"surcharge_level":[2]})
    part=pd.DataFrame({"event_id":[1,1,1],"client_id":["C001","C002","C003"],"response":["accept","decline","no_response"]})
    out=event_table(events,part).iloc[0]; assert out.invited==3; assert out.participation_pct==pytest.approx(100/3)


def test_household_summary_uses_operational_energy():
    frame=pd.DataFrame({"aggregate_power_kw":[1.,2.],"aggregate_power_energy_kwh":[.5,1.]})
    s=summary(frame); assert s["energy_kwh"]==1.5; assert s["load_factor"]==.75
