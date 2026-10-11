"""Chat assistant, confirmation deadline and the demo service behind the Streamlit pages."""
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.orchestration.lifecycle import Orchestrator
from backend.orchestration.notifications.assistant import ChatContext, detect_intent, reply


@pytest.mark.parametrize("text,intent", [
    ("When is the peak?", "when"), ("quel est le prix ?", "price"), ("Que faire ?", "action"),
    ("combien je vais payer", "cost"), ("pourquoi une pointe", "why"), ("ok", "accept"), ("non merci", "decline"),
    ("plus tard", "later"), ("لاحقا", "later"), ("لا", "decline"), ("كم سأدفع", "cost"), ("متى الذروة", "when"),
    ("how do I compare with others", "compare"), ("mon historique", "history"), ("I booked a table", "help"),
])
def test_intents(text, intent):
    assert detect_intent(text) == intent


EV = SimpleNamespace(event_id=1, start_ts=pd.Timestamp("2025-07-15 18:00"), end_ts=pd.Timestamp("2025-07-15 20:00"),
                     level=2, deficit_share=0.052)
TG = SimpleNamespace(exempt=False, band_price=218, peak_price=327, personalised=True, action="reduce_ac",
                     expected_saving_dt=0.15, usual_window_kwh=2.75, expected_surcharge_dt=0.30, comparison_message=None)


def _ctx(now="2025-07-14 19:00", lang="en", event=EV, target=TG):
    return ChatContext(now=pd.Timestamp(now), lang=lang, event=event, target=target, last_bill=None,
                       n_events=3, n_accepted=1, total_surcharge=0.9, comparison_message=None)


def test_answers_use_stored_numbers():
    assert "327 millimes/kWh instead of 218" in reply("price?", _ctx())[1]
    assert "0.15 DT" in reply("what should I do", _ctx())[1]
    assert "0,30 DT" in reply("combien ?", _ctx(lang="fr"))[1]
    assert "5.2%" in reply("why", _ctx())[1]
    assert "no peak event" in reply("price?", _ctx(event=None, target=None))[1]


def test_chat_response_is_recorded_until_the_event_starts():
    saved = []
    reply("ok", _ctx(), saved.append)
    assert saved == ["accept"]
    intent, text = reply("plus tard", _ctx(now="2025-07-15 18:30", lang="fr"), saved.append)
    assert intent == "later" and saved == ["accept"] and "déjà commencé" in text


def test_confirm_deadline_is_the_alert_time_the_day_before():
    assert Orchestrator.confirm_deadline("2025-07-15 18:00") == pd.Timestamp("2025-07-14 18:00")


def _community_model_available():
    try:
        from backend.intelligence.model_registry import MODEL_REGISTRY
        return MODEL_REGISTRY.community_available()
    except Exception:
        return False


@pytest.mark.skipif(not _community_model_available(), reason="community DR model artifact not installed")
def test_demo_service_flow(tmp_path, monkeypatch):
    from backend.services import orchestrator_service as svc
    monkeypatch.setattr(svc, "APP_DB", tmp_path / "app.db")
    monkeypatch.setattr(svc, "CLOCK_FILE", tmp_path / "clock.json")
    svc._session.cache_clear()
    svc.orchestrator.cache_clear()
    try:
        assert svc.build_history("2025-06-28") == pd.Timestamp("2025-06-27 23:30")
        assert svc.jump_to_next_peak() == pd.Timestamp("2025-07-01 14:30")           # peak forecast for 2 July
        ev = svc.events()
        eid = int(ev[ev.status == "proposed"].event_id.iloc[0])
        svc.confirm(eid)
        assert len(svc.targets(eid)) == 50
        assert svc.next_step()[0] == pd.Timestamp("2025-07-01 18:00")                # alerts
        cid = svc.targets(eid).client_id.iloc[0]
        event, target = svc.current_event(cid)
        assert event.event_id == eid and len(svc.notifications(cid)) >= 1
        intent, _ = svc.chat(cid, "ok")
        assert intent == "accept" and svc.targets(eid).set_index("client_id").loc[cid, "response"] == "accept"
        svc.advance_to("2025-07-03 09:00")                                            # event, then bills
        hist = svc.household_events(cid)
        assert hist.set_index("event_id").loc[eid, "surcharge_dt"] >= 0
        assert svc.events().set_index("event_id").loc[eid, "status"] == "billed"
    finally:
        svc._session.cache_clear()
        svc.orchestrator.cache_clear()
