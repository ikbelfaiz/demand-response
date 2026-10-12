"""Customer profiling: train (January-May), replay the historical events, evaluate.

    python scripts/train_customer_profiling.py                       # outputs
    python scripts/train_customer_profiling.py --evaluate            # + performance tests
    python scripts/train_customer_profiling.py --train-end 2025-06-16  # January to mid-June

Outputs -> models/customer_profiling/kmeans_v1/
  household_segments.csv     (1) segments, acceptance rate, comparison      -> interface
  segment_load_profiles.csv  (2) curves by segment, season, day type, slot  -> interface
  event_targeting.csv        (3) priority list per event                    -> event manager
  chatbot_context.csv        (4) context per household and event            -> chatbot
  event_costs.csv            (5a) cost of each event per household           -> pricing, interface
  segment_savings.csv        (5b) savings per segment and event              -> pricing, dashboard
  rewards.csv                (5c) rewards of accepting households            -> pricing
  metadata.json              the saved model
Performance report -> outputs/customer_profiling/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import PROJECT_ROOT  # noqa: E402
from backend.data import load_events, load_grid, load_household, load_household_info, load_participation  # noqa: E402
from backend.intelligence import customer_profiling as cp  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "models" / "customer_profiling" / "kmeans_v1"
REPORT_DIR = PROJECT_ROOT / "outputs" / "customer_profiling"
TEST_START, TEST_END = pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-01")   # June-September events


def load_half_hour_meters(client_ids: list[str]) -> tuple[pd.DataFrame, int]:
    series, spikes = {}, 0
    for i, cid in enumerate(client_ids, 1):
        clean, n = cp.clean_household_power(load_household(cid, columns=["aggregate_power_w"]))
        series[cid], spikes = cp.to_half_hour_kw(clean), spikes + n
        print(f"\r  meters {i}/{len(client_ids)}", end="", flush=True)
    print()
    return pd.DataFrame(series), spikes


def half_hour_grid(index: pd.DatetimeIndex) -> pd.DataFrame:
    g = load_grid().set_index("timestamp")
    r = g.resample(cp.SLOT)
    out = pd.DataFrame({"temperature_c": r["temperature_c"].mean(), "is_dr_event": r["is_dr_event"].max(),
                        "is_dr_peak": r["is_dr_peak"].max(),
                        "deficit_kw": (g["households_consumption_kw"] - g["steg_production_kw"] - g["pv_production_kw"])
                        .clip(lower=0).resample(cp.SLOT).mean()})
    return out.reindex(index)


def replay_events(kw30, grid30, segments, events, participation):
    """Before-event outputs for every historical event, with data available at D-1 14:00 only.

    The observed deficit in the event window stands in for the forecast deficit.
    """
    targeting, context = [], []
    for ev in events.itertuples():
        as_of = ev.start_ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        acc = cp.acceptance_rates(participation, events, as_of)
        deficit = float(cp._window(grid30["deficit_kw"], ev.start_ts, ev.end_ts).sum() * cp.SLOT_HOURS)
        t = cp.event_targeting(segments, kw30, grid30, acc, ev.start_ts, ev.end_ts, deficit, as_of)
        c = cp.chatbot_context(segments, kw30, grid30, acc, ev.start_ts, ev.end_ts, as_of)
        targeting.append(t.assign(event_id=ev.event_id, as_of=as_of))
        context.append(c.assign(event_id=ev.event_id, as_of=as_of))
    first = lambda df: df[["event_id"] + [c for c in df.columns if c != "event_id"]]  # noqa: E731
    return first(pd.concat(targeting, ignore_index=True)), first(pd.concat(context, ignore_index=True))


def economics(kw30, segments, events, participation):
    costs = pd.concat([cp.event_costs(kw30, ev, participation, surcharge_level=int(ev.surcharge_level))
                       for ev in events.itertuples()], ignore_index=True)
    savings = cp.segment_savings(kw30, segments, events, participation)
    return costs, savings, cp.rewards(savings, segments, participation, costs)


def evaluate(feats, labels, model, segments, kw30, grid30, events, participation, targeting, savings, args):
    from sklearn.metrics import adjusted_rand_score
    from backend.intelligence import evaluation as ev_

    test_ids = events.loc[(events.start_ts >= max(TEST_START, args.train_end)) & (events.start_ts < TEST_END), "event_id"].tolist()
    # stability: same clustering on June-December
    later = cp.compute_features(kw30, grid30, pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01"))
    ari = adjusted_rand_score(labels, cp.kmeans_labels(later.reindex(feats.index), args.k))
    # meaningful segments
    info = load_household_info()
    meaning = segments.merge(info, on="client_id").groupby("segment")[
        ["n_occupants", "has_ac", "has_electric_water_heater", "occupied_daytime"]].mean()
    # share of peak consumption (is_dr_peak slots, June-September)
    peak = cp._window(kw30, TEST_START, TEST_END).loc[cp._window(grid30, TEST_START, TEST_END)["is_dr_peak"] > 0]
    peak_share = (peak.sum() * cp.SLOT_HOURS).groupby(segments.set_index("client_id")["segment"]).sum()
    value = ev_.targeting_value(segments, test_ids)
    value["share_of_peak_consumption"] = peak_share / peak_share.sum()
    value["share_of_households"] = segments.segment.value_counts(normalize=True)
    report = {
        "train_window": [str(args.train_start.date()), str((args.train_end - pd.Timedelta(days=1)).date())],
        "hot_days_in_train_window": feats.attrs.get("hot_days"),
        "test_events": test_ids,
        "cluster_quality_silhouette": round(model.silhouette, 3),
        "stability_ari_train_vs_jun_dec": round(float(ari), 3),
        "economics_test_events": ev_.evaluate_segment_savings(savings, test_ids),
        "economics_full_year": ev_.evaluate_segment_savings(savings),
        "priority_list_top_50pct": ev_.priority_list_value(targeting, test_ids),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "performance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    meaning.to_csv(REPORT_DIR / "meaningful_segments.csv")
    value.to_csv(REPORT_DIR / "targeting_value.csv")
    ev_.segment_savings_error(savings, segments, test_ids).to_csv(REPORT_DIR / "segment_savings_error.csv")
    print("\nPerformance report:\n" + json.dumps(report, indent=2))
    print("\nMeaningful segments:\n" + meaning.round(2).to_string())
    print("\nValue for targeting (June-September):\n" + value.round(3).to_string())
    print("\nSavings error per segment (test events):\n" +
          ev_.segment_savings_error(savings, segments, test_ids).round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=cp.N_CLUSTERS)
    ap.add_argument("--train-start", type=pd.Timestamp, default=cp.TRAIN_START)
    ap.add_argument("--train-end", type=pd.Timestamp, default=cp.TRAIN_END, help="exclusive")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for old in ARTIFACT_DIR.glob("*.csv"):
        old.unlink()

    info, participation = load_household_info(), load_participation()
    events = load_events().sort_values("start_ts", ignore_index=True)
    print("Meters -> 30 minutes")
    kw30, spikes = load_half_hour_meters(info.client_id.tolist())
    grid30 = half_hour_grid(kw30.index)
    as_of = kw30.index.max() + pd.Timedelta(cp.SLOT)

    feats = cp.compute_features(kw30, grid30, args.train_start, args.train_end)
    model, labels = cp.fit(feats, args.k)
    acc_now = cp.acceptance_rates(participation, events, as_of)
    segments = cp.household_segments(feats, labels, model, acc_now, kw30, grid30, as_of)
    profiles = cp.segment_load_profiles(segments, kw30, grid30)
    print("Replaying the historical events")
    targeting, context = replay_events(kw30, grid30, segments, events, participation)
    costs, savings, rewards = economics(kw30, segments, events, participation)

    outputs = {"household_segments": segments, "segment_load_profiles": profiles, "event_targeting": targeting,
               "chatbot_context": context, "event_costs": costs, "segment_savings": savings, "rewards": rewards}
    for name, frame in outputs.items():
        frame.to_csv(ARTIFACT_DIR / f"{name}.csv", index=False)
    metadata = {"model_name": "customer_profiling_kmeans", "model_version": "v1", "k": args.k,
                "frequency": cp.SLOT, "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "train_start": str(args.train_start.date()), "train_end_exclusive": str(args.train_end.date()),
                "hot_days_in_train_window": feats.attrs["hot_days"], "as_of": str(as_of),
                "n_households": int(len(segments)), "n_events_replayed": int(events.shape[0]),
                "spikes_removed": int(spikes), **model.to_dict()}
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nTrain {args.train_start.date()} -> {args.train_end.date()} (excl.), "
          f"{feats.attrs['hot_days']} hot days, silhouette {model.silhouette:.3f}")
    print(segments.groupby("segment").agg(households=("client_id", "size"), priority=("dr_priority", "first"),
                                          annual_kwh=("annual_kwh", "mean"), evening_share=("evening_share", "mean"),
                                          acceptance=("acceptance_rate", "mean")).sort_values("priority").round(2).to_string())
    if args.evaluate:
        evaluate(feats, labels, model, segments, kw30, grid30, events, participation, targeting, savings, args)
    print(f"\nDone in {time.time() - t0:.0f}s -> {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
