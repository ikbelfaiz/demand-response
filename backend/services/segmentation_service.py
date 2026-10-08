"""Transparent rule-based segmentation (not trained clustering)."""
import pandas as pd


def consumption_quantiles(energy: pd.Series) -> pd.Series:
    labels = ["Lower-use", "Typical-use", "Higher-use", "Highest-use"]
    if energy.nunique() < 4:
        return pd.Series("Typical-use", index=energy.index)
    return pd.qcut(energy.rank(method="first"), 4, labels=labels)
