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



def _truth() -> pd.DataFrame:
    return load_ground_truth_for_evaluation().assign(true_savings_kwh=lambda x: x.true_baseline_kwh - x.actual_kwh)


def evaluate_segment_savings(estimates: pd.DataFrame, event_ids=None) -> dict:
    """Economics test: error of the comparison-group savings vs synthetic truth (accepting households)."""
    t = _truth().loc[lambda x: x.response.eq("accept")]
    if event_ids is not None:
        t = t.loc[t.event_id.isin(event_ids)]
        estimates = estimates.loc[estimates.event_id.isin(event_ids)]
    true_ev = t.groupby("event_id").agg(true_savings_kwh=("true_savings_kwh", "sum"),
                                        true_baseline_kwh=("true_baseline_kwh", "sum"))
    j = estimates.groupby("event_id")[["savings_kwh", "expected_kwh"]].sum().join(true_ev, how="inner")
    err = (j.savings_kwh - j.true_savings_kwh).abs() / j.true_baseline_kwh
    return {"n_events": int(len(j)),
            "true_savings_kwh": round(float(j.true_savings_kwh.sum()), 1),
            "estimated_savings_kwh": round(float(j.savings_kwh.sum()), 1),
            "total_error_pct": round(float(100 * (j.savings_kwh.sum() / j.true_savings_kwh.sum() - 1)), 1),
            "mean_abs_error_per_event_pct_of_no_dr": round(float(100 * err.mean()), 1),
            "true_reduction_pct": round(float(100 * j.true_savings_kwh.sum() / j.true_baseline_kwh.sum()), 1),
            "estimated_reduction_pct": round(float(100 * j.savings_kwh.sum() / j.expected_kwh.sum()), 1)}


def segment_savings_error(estimates: pd.DataFrame, segments: pd.DataFrame, event_ids) -> pd.DataFrame:
    """Economics test per segment: estimated vs true savings over the test events."""
    t = _truth().loc[lambda x: x.response.eq("accept") & x.event_id.isin(event_ids)]
    t = t.merge(segments[["client_id", "segment"]], on="client_id")
    true = t.groupby("segment")["true_savings_kwh"].sum()
    est = estimates.loc[estimates.event_id.isin(event_ids)].groupby("segment")["savings_kwh"].sum()
    out = pd.DataFrame({"true_savings_kwh": true, "estimated_savings_kwh": est})
    out["error_pct"] = 100 * (out.estimated_savings_kwh / out.true_savings_kwh - 1)
    return out


def targeting_value(segments: pd.DataFrame, event_ids) -> pd.DataFrame:
    """Value for targeting: true kWh saved per accepted event, by segment."""
    t = _truth().loc[lambda x: x.event_id.isin(event_ids)].merge(segments[["client_id", "segment"]], on="client_id")
    acc = t.loc[t.response.eq("accept")]
    out = acc.groupby("segment").agg(accepted_events=("event_id", "size"),
                                     kwh_saved_per_accepted_event=("true_savings_kwh", "mean"),
                                     true_savings_kwh=("true_savings_kwh", "sum"))
    out["acceptance_rate"] = t.groupby("segment")["response"].apply(lambda r: r.eq("accept").mean())
    return out


def priority_list_value(targeting: pd.DataFrame, event_ids, top_share: float = 0.5) -> dict:
    """Share of the total true savings obtained by notifying only the top of the priority list."""
    t = _truth().loc[lambda x: x.event_id.isin(event_ids)]
    shares = []
    for ev in event_ids:
        lst = targeting.loc[targeting.event_id.eq(ev)].sort_values("rank")
        top = set(lst.client_id.head(int(round(len(lst) * top_share))))
        e = t.loc[t.event_id.eq(ev)]
        total = e.true_savings_kwh.sum()
        if total > 0:
            shares.append(e.loc[e.client_id.isin(top), "true_savings_kwh"].sum() / total)
    return {"top_share_of_list": top_share, "n_events": len(shares),
            "share_of_true_savings_pct": round(100 * float(pd.Series(shares).mean()), 1),
            "random_list_expectation_pct": round(100 * top_share, 1)}
