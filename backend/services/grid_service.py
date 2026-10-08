"""Modeled neighborhood supply-demand calculations (all values in kW)."""
import pandas as pd


def supply_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["available_supply_kw"] = out["steg_production_kw"] + out["pv_production_kw"]
    out["margin_kw"] = out["available_supply_kw"] - out["households_consumption_kw"]
    return out


PEAK_ASSUMPTION = (
    "Synthetic generator label: a centered 15-minute mean of feeder demand (already including "
    "the 3% modeled network loss) exceeds calibrated available STEG + PV allocation."
)
