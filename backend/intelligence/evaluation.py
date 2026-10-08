"""Research-only evaluation boundary for synthetic counterfactual savings."""
import pandas as pd
from backend.data.v3_loader import load_ground_truth_for_evaluation


def evaluate_savings(estimates: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "client_id", "estimated_savings_kwh"}
    if not required.issubset(estimates): raise ValueError(f"Missing fields: {sorted(required - set(estimates))}")
    truth = load_ground_truth_for_evaluation().assign(
        true_savings_kwh=lambda x: x.true_baseline_kwh - x.actual_kwh)
    out = estimates.merge(truth[["event_id", "client_id", "true_savings_kwh"]], on=["event_id", "client_id"])
    out["error_kwh"] = out.estimated_savings_kwh - out.true_savings_kwh
    return out
