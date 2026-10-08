"""Supply forecast semantics shared by replay and reporting."""
from __future__ import annotations

import pandas as pd


def combine_supply_forecasts(predicted_steg_kw, predicted_pv_kw) -> pd.DataFrame:
    frame = pd.DataFrame({"predicted_steg_kw": predicted_steg_kw, "predicted_pv_kw": predicted_pv_kw})
    frame["predicted_supply_kw"] = frame.predicted_steg_kw + frame.predicted_pv_kw
    return frame
