"""DR orchestrator: decision on top of the team's recommendations, messages, and a two-week replay."""
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.intelligence.orchestrator import status
from backend.orchestration.decide import choose_level, decide
from backend.orchestration.notifications.render import render_for_target


def make_forecast(supply=100.0):
    ts = pd.date_range("2025-07-15", periods=48, freq="30min")
    return pd.DataFrame({"timestamp": ts, "predicted_demand_kw": 80.0, "predicted_supply_kw": supply,
                         "predicted_margin_kw": supply - 80.0})


def make_recs(rows):
    return pd.DataFrame([{"start_ts": pd.Timestamp(s), "end_ts": pd.Timestamp(e), "max_deficit_kw": d,
                          "shortage_energy_kwh": k, "recommended_reduction_pct": 5.0} for s, e, d, k in rows])


def test_no_recommendation_no_event():
    assert decide(make_forecast(), make_recs([]), 0) is None


def test_keeps_largest_shortage_and_sets_level():
    recs = make_recs([("2025-07-15 18:00", "2025-07-15 19:00", 3.0, 2.0),
                      ("2025-07-15 20:00", "2025-07-15 22:00", 9.0, 16.0)])
    d = decide(make_forecast(), recs, 0)
    assert d.start == pd.Timestamp("2025-07-15 20:00") and d.end == pd.Timestamp("2025-07-15 22:00")
    assert d.deficit_share == 0.08 and d.level == 2      # 16 kWh / (4 slots × 100 kW × 0.5 h) = 8%


def test_weekly_limit():
    recs = make_recs([("2025-07-15 20:00", "2025-07-15 21:00", 5.0, 5.0)])
    assert decide(make_forecast(), recs, 3) is None


def test_levels():
    assert choose_level(0.04) == 1 and choose_level(0.10) == 2 and choose_level(0.20) == 3
    assert choose_level(0.04, outage=True) == 3


def test_status_reports_orchestration():
    st = status()
    assert st.notification_engine and st.pricing_engine and not st.operational_activation


def test_alert_matches_the_designed_card():
    ev = SimpleNamespace(start_ts=pd.Timestamp("2025-07-16 18:00"), end_ts=pd.Timestamp("2025-07-16 20:00"))
    t = SimpleNamespace(peak_price=327, band_price=218, action="reduce_ac",
                        expected_saving_dt=0.147, expected_surcharge_dt=0.349)
    text, _ = render_for_target("alert_personal", "en", ev, t, level_pct=50)
    assert text == ("Tomorrow between 6 PM and 8 PM, critical peak pricing will be active.\n"
                    "Peak price: 327 millimes/kWh (+50% vs STEG price of 218 millimes)\n"
                    "Recommended action: reduce your air conditioner usage\n"
                    "Expected saving: ~0.15 DT if you reduce")


def test_all_languages_render():
    ev = SimpleNamespace(start_ts=pd.Timestamp("2025-07-16 18:00"), end_ts=pd.Timestamp("2025-07-16 20:30"))
    t = SimpleNamespace(peak_price=327, band_price=218, action="shift_water_heater",
                        expected_saving_dt=0.16, expected_surcharge_dt=0.349)
    for lang in ("en", "fr", "ar"):
        for kind in ("alert_personal", "alert_general", "alert_exempt", "cancellation"):
            text, _ = render_for_target(kind, lang, ev, t, level_pct=50)
            assert "{{" not in text and len(text) > 20


def _community_model_available():
    from backend.intelligence.model_registry import MODEL_REGISTRY
    return MODEL_REGISTRY.community_available()


@pytest.mark.skipif(not _community_model_available(), reason="community DR model artifact not installed")
def test_replay_two_weeks_with_team_forecast(tmp_path, monkeypatch):
    from backend.orchestration import settings
    from backend.orchestration.replay import run
    monkeypatch.setattr(settings, "OUT_DIR", tmp_path)
    r = run("2025-07-01", "2025-07-14", db_url=f"sqlite:///{tmp_path / 'r.db'}", quiet=True)
    ev, tg, nt, eb = r["events"], r["targets"], r["notifications"], r["event_bills"]
    assert len(ev) >= 1
    assert (ev.forecast_model_version != "").all() and (ev.decision_policy_version != "").all()
    assert len(tg) == len(ev[ev.status != "vetoed"]) * 50                 # every household informed
    assert len(eb) == len(ev[ev.status == "billed"]) * 50 and (eb.surcharge_dt >= 0).all()
    assert nt[nt.kind == "reminder"].groupby(["event_id", "client_id"]).size().max() <= 1
