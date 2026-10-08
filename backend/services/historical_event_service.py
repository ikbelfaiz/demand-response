import pandas as pd

from backend.utils.time_utils import event_windows


def historical_event_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Return supplied labels for retrospective exploration, never as predictions."""
    if "dr_event" not in data:
        return pd.DataFrame(columns=["start", "end", "duration_minutes", "peak_household_kw", "peak_zone_mw"])
    rows = []
    for start, end in event_windows(data, "dr_event"):
        event = data.loc[(data.timestamp >= start) & (data.timestamp < end)]
        rows.append({"start": start, "end": end, "duration_minutes": int((end-start).total_seconds()/60),
                     "peak_household_kw": float(event.household_power_kw.max()),
                     "peak_zone_mw": float(event.zone_demand_mw.max())})
    return pd.DataFrame(rows)

