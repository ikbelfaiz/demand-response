import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.intelligence import customer_profiling as cp
from backend.services import segmentation_service as seg

KINDS = {"large": (0.3, 1.6, 0.15), "day": (0.6, 0.5, 0.09), "med": (0.25, 0.6, 0.11), "low": (0.28, 0.5, 0.0)}


@pytest.fixture(scope="module")
def world():
    """Four clearly separated household types over 2025, three summer events."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", "2025-12-31 23:30", freq="30min")
    h = np.asarray(idx.hour + idx.minute / 60); doy = np.asarray(idx.dayofyear)
    temp = 19 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + 6 * np.sin(2 * np.pi * (h - 9) / 24)
    cols, kinds = {}, []
    for kind, n in zip(KINDS, (6, 3, 5, 4)):
        base, eve, ac = KINDS[kind]
        for _ in range(n):
            p = base + eve * np.exp(-((h - 20) ** 2) / 4) + ac * np.clip(temp - 24, 0, None)
            if kind == "day": p = p + 0.4 * ((h > 9) & (h < 17))
            cols[f"C{len(cols) + 1:03d}"] = p * rng.lognormal(0, 0.05, len(idx)); kinds.append(kind)
    kw30 = pd.DataFrame(cols, index=idx)
    days = pd.DatetimeIndex(["2025-07-10", "2025-07-20", "2025-08-05"])
    events = pd.DataFrame({"event_id": [1, 2, 3], "start_ts": days + pd.Timedelta(hours=19),
                           "end_ts": days + pd.Timedelta(hours=21), "surcharge_level": 2})
    is_ev = (np.isin(idx.normalize(), days) & (h >= 19) & (h < 21)).astype(int)
    grid30 = pd.DataFrame({"temperature_c": temp, "is_dr_event": is_ev}, index=idx)
    resp = ["accept" if i % 2 == 0 else "decline" for i in range(len(cols))]
    part = pd.concat([pd.DataFrame({"event_id": e, "client_id": list(cols), "response": resp}) for e in events.event_id])
    reduced = kw30.copy()
    for e in events.itertuples():                                   # accepters cut 20 %
        m = (reduced.index >= e.start_ts) & (reduced.index < e.end_ts)
        reduced.loc[m, list(cols)[::2]] *= 0.8
    feats = cp.compute_features(reduced, grid30)
    model, labels = cp.fit(feats)
    as_of = pd.Timestamp("2026-01-01")
    acc = cp.acceptance_rates(part, events, as_of)
    segments = cp.household_segments(feats, labels, model, acc, reduced, grid30, as_of)
    return SimpleNamespace(kw30=reduced, grid30=grid30, kinds=pd.Series(kinds, index=list(cols)), events=events,
                           part=part, feats=feats, model=model, labels=labels, acc=acc, segments=segments)


def test_cleaning_and_half_hour_resampling():
    ts = pd.date_range("2025-01-01", periods=60, freq="1min")
    raw = pd.DataFrame({"timestamp": ts, "aggregate_power_w": [2000.0] * 60})
    raw.loc[3, "aggregate_power_w"] = 20_000; raw.loc[7, "aggregate_power_w"] = np.nan
    clean, spikes = cp.clean_household_power(raw)
    assert spikes == 1 and clean.notna().all()
    assert np.allclose(cp.to_half_hour_kw(clean), 2.0)


def test_features_use_train_window_only(world):
    assert list(world.feats.columns) == cp.FEATURES
    late = world.kw30.copy(); late.loc[late.index >= cp.TRAIN_END] *= 10
    assert np.allclose(cp.compute_features(late, world.grid30).annual_kwh, world.feats.annual_kwh)


def test_segments_named_correctly(world):
    s = world.segments.set_index("client_id").segment.reindex(world.kinds.index)
    expected = world.kinds.map({"large": cp.SEG_LARGE, "day": cp.SEG_DAYTIME, "med": cp.SEG_MEDIUM, "low": cp.SEG_LOW})
    assert (s == expected).all()


def test_model_roundtrip(world):
    assert (cp.ProfilingModel.from_dict(world.model.to_dict()).predict(world.feats) == world.labels).all()


def test_acceptance_rate_updated_after_each_event(world):
    before_second = cp.acceptance_rates(world.part, world.events, world.events.end_ts.iloc[0])
    assert before_second.n_invitations.eq(1).all()
    assert world.acc.n_invitations.eq(3).all() and set(world.acc.acceptance_rate.round(2)) == {0.0, 1.0}


def test_output1_has_segment_acceptance_and_comparison(world):
    cols = {"client_id", "segment", "dr_priority", "acceptance_rate", "vs_similar_households_pct", "comparison_message"}
    assert cols <= set(world.segments.columns)


def test_output2_curves_by_segment_season_daytype_slot(world):
    load = cp.segment_load_profiles(world.segments, world.kw30, world.grid30)
    assert len(load) == 4 * 4 * 2 * 48 and set(load.day_type) == {"weekday", "weekend"}


def test_output3_targeting_order_and_deficit_coverage(world):
    t = cp.event_targeting(world.segments, world.kw30, world.grid30, world.acc,
                           "2025-08-20 19:00", "2025-08-20 21:00", 2.0, "2025-08-19 14:00")
    assert t.dr_priority.is_monotonic_increasing and t.iloc[0].segment == cp.SEG_LARGE
    for _, g in t.groupby("dr_priority"):
        assert g.usual_window_kwh.is_monotonic_decreasing
    assert t.cumulative_expected_reduction_kwh.is_monotonic_increasing


def test_output3_uses_only_data_before_as_of(world):
    future = world.kw30.copy(); future.loc[future.index >= "2025-08-19 14:00"] *= 50
    args = (world.segments, world.grid30, world.acc, "2025-08-20 19:00", "2025-08-20 21:00", 2.0, "2025-08-19 14:00")
    a = cp.event_targeting(args[0], world.kw30, *args[1:])
    b = cp.event_targeting(args[0], future, *args[1:])
    assert np.allclose(a.usual_window_kwh, b.usual_window_kwh)


def test_output4_chatbot_context(world):
    c = cp.chatbot_context(world.segments, world.kw30, world.grid30, world.acc,
                           "2025-08-20 19:00", "2025-08-20 21:00", "2025-08-19 14:00")
    assert {"segment", "usual_window_kwh", "acceptance_rate", "steg_band_millimes",
            "comparison_message"} <= set(c.columns) and len(c) == 18


@pytest.mark.parametrize("kwh,price", [(50, 62), (100, 96), (200, 176), (300, 218), (301, 341), (500, 341), (501, 414)])
def test_steg_band(kwh, price):
    assert cp.steg_band(kwh) == price


def test_output5a_event_cost_and_exemptions(world):
    ev = world.events.iloc[0]
    costs = cp.event_costs(world.kw30, ev, world.part, control_group=["C002"]).set_index("client_id")
    informed = costs.loc["C001"]
    assert informed.event_cost_dt == pytest.approx(informed.window_kwh * informed.projected_band_millimes * 1.5 / 1000)
    assert costs.loc["C002", "surcharge_dt"] == 0                       # control group exempt
    tiny = world.kw30 * 0.01                                            # economic band never surcharged
    assert cp.event_costs(tiny, ev, world.part).surcharge_dt.eq(0).all()


def test_output5b_savings_recover_known_reduction(world):
    s = cp.segment_savings(world.kw30, world.segments, world.events, world.part)
    assert 100 * s.savings_kwh.sum() / s.expected_kwh.sum() == pytest.approx(20, abs=3)


def test_output5b_control_group_is_used_when_set(world):
    controls = {1: ["C002", "C004"]}
    s = cp.segment_savings(world.kw30, world.segments, world.events.iloc[:1], world.part, control_groups=controls)
    assert set(s.comparison_source) <= {"control_group", "event_fallback"}


def test_output5c_rewards_only_for_accepters(world):
    costs = pd.concat([cp.event_costs(world.kw30, e, world.part) for e in world.events.itertuples()])
    s = cp.segment_savings(world.kw30, world.segments, world.events, world.part)
    r = cp.rewards(s, world.segments, world.part, costs)
    accepters = set(world.part.loc[world.part.response.eq("accept"), "client_id"])
    assert set(r.client_id) <= accepters and (r.reward_dt >= 0).all()


def test_profiling_never_reads_ground_truth():
    assert "load_ground_truth" not in inspect.getsource(cp)


@pytest.mark.skipif(not seg.profiling_available(), reason="profiling outputs not trained")
def test_installed_outputs_are_consistent():
    hh, meta = seg.household_segments(), seg.load_metadata()
    assert len(hh) == meta["n_households"] == 50 and hh.segment.nunique() == meta["k"]
    assert len(seg.event_targeting(event_id=30)) == 50
    assert seg.chatbot_context("C001", event_id=30)["segment"]
    assert not seg.event_costs(event_id=30).empty and not seg.segment_savings(event_id=30).empty
