"""Operational baseline interface deliberately isolated from synthetic truth."""
import pandas as pd


BASELINE_COLUMNS = ["timestamp", "estimated_baseline_kw", "method", "uncertainty_kw"]


def estimate_baseline(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError("Baseline estimation is planned but not implemented.")
