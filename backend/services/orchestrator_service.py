"""DR orchestrator service for the Streamlit pages (demo clock, history, queries, household actions).

The dataset is historical, so the app runs the orchestrator on a DEMO CLOCK:
  * build_history(until)   replays the orchestrator from HISTORY_START to `until` with simulated household
                           answers (acceptance rate of the customer profiling model) → past events, bills.
  * advance_to(t) / next_step() / jump_to_next_peak()
                           move the clock forward in 30-min ticks with NO simulation: proposals wait for the
                           operator (Confirm / Veto) and answers come from the consumer page.
State lives in outputs/orchestrator/app.db (SQLite) and app_clock.json, shared by both perspectives.
"""
from __future__ import annotations

import json
from collections import deque
from functools import lru_cache

import pandas as pd
from sqlalchemy import select

from backend.orchestration import settings
from backend.orchestration.store import (AppliancePreference, Client, DREvent, EventBill, EventTarget, MonthlyBill,
                                         Notification, UserPreference, get_engine, get_sessionmaker, init_db)

APP_DB = settings.OUT_DIR / "app.db"
CLOCK_FILE = settings.OUT_DIR / "app_clock.json"
HISTORY_START = pd.Timestamp("2025-06-01")
DEFAULT_HISTORY_END = pd.Timestamp("2025-08-01")
DATA_END = pd.Timestamp("2025-12-31 23:30")
LOG = deque(maxlen=200)                     # orchestrator log lines shown on the operator page
STEP = pd.Timedelta(settings.SLOT)
LANGUAGES = {"fr": "Français", "ar": "العربية", "en": "English"}
DEVICES = {"ac": "Air conditioner", "water_heater": "Water heater", "washing_machine": "Washing machine"}


def _url() -> str:
    return f"sqlite:///{APP_DB}"


def _log(line: str) -> None:
    LOG.appendleft(line)


# ------------------------------------------------------------------ orchestrator + clock
@lru_cache(maxsize=1)
def _session():
    settings.OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine(_url()) if APP_DB.is_file() else init_db(_url())
    return get_sessionmaker(engine)


@lru_cache(maxsize=1)
def orchestrator():
    """Live orchestrator: no auto-confirm, no simulated answers. Built once per process (loads the models)."""
    from backend.orchestration.lifecycle import Orchestrator
    return Orchestrator(_session(), auto_confirm=False, response_simulator=None, log=_log)


def initialised() -> bool:
    return APP_DB.is_file() and CLOCK_FILE.is_file()


def clock() -> pd.Timestamp | None:
    try:
        return pd.Timestamp(json.loads(CLOCK_FILE.read_text(encoding="utf-8"))["now"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _set_clock(now: pd.Timestamp) -> None:
    CLOCK_FILE.write_text(json.dumps({"now": str(now)}), encoding="utf-8")


def build_history(until=DEFAULT_HISTORY_END, start=HISTORY_START, progress=None) -> pd.Timestamp:
    """Reset the demo database and replay the orchestrator from `start` to `until` (simulated answers)."""
    from backend.orchestration.lifecycle import Orchestrator
    from backend.orchestration.replay import make_response_simulator
    live = orchestrator()
    engine = init_db(_url())                       # drop + recreate + seed
    live.Session.configure(bind=engine)
    sim = Orchestrator(live.Session, live.forecaster, live.recommender, live.targeting, grid30=live.grid30,
                       meter30=live.meter30, households=live.households, auto_confirm=True,
                       response_simulator=make_response_simulator(), log=_log)
    until = pd.Timestamp(until).floor(settings.SLOT)
    ticks = pd.date_range(pd.Timestamp(start), until - STEP, freq=settings.SLOT)
    for i, now in enumerate(ticks):
        sim.tick(now)
        if progress and i % 96 == 0:
            progress(i / max(len(ticks), 1))
    _set_clock(until - STEP)
    _log(f"{until:%Y-%m-%d %H:%M} history built from {start:%Y-%m-%d} (simulated household answers)")
    return clock()


def advance_to(target, progress=None) -> pd.Timestamp:
    """Run every 30-min tick after the current clock up to `target` (included)."""
    now = clock()
    target = min(pd.Timestamp(target).floor(settings.SLOT), DATA_END)
    if now is None or target <= now:
        return now
    orch = orchestrator()
    ticks = pd.date_range(now + STEP, target, freq=settings.SLOT)
    for i, t in enumerate(ticks):
        orch.tick(t)
        if progress and i % 48 == 0:
            progress(i / max(len(ticks), 1))
    _set_clock(target)
    return target


def _daily(now: pd.Timestamp, hm: str) -> pd.Timestamp:
    t = now.normalize() + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))
    return t if t > now else t + pd.Timedelta(days=1)


def next_milestone(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, str]:
    """Next time at which the orchestrator does something visible."""
    now = now or clock()
    cands = [(_daily(now, settings.DECISION_TIME), "D-1 decision (forecast → proposed event)"),
             (_daily(now, settings.ALERT_TIME), "alerts sent to households"),
             (_daily(now, settings.LATER_TIME), "second alert for \"Later\""),
             (_daily(now, settings.BILLING_TIME), "bills and summaries of yesterday's event")]
    with _session()() as s:
        for ev in s.scalars(select(DREvent).where(DREvent.status.in_(["confirmed", "notified", "active"]))):
            st, en = pd.Timestamp(ev.start_ts), pd.Timestamp(ev.end_ts)
            for t, label in [(st, f"event {ev.event_id} starts"), (st + STEP, f"event {ev.event_id}: first-slot check"),
                             (en, f"event {ev.event_id} ends")]:
                if t > now:
                    cands.append((t, label))
    return min(cands, key=lambda c: c[0])


def next_step(progress=None) -> tuple[pd.Timestamp, str]:
    t, label = next_milestone()
    advance_to(t, progress)
    return t, label


def jump_to_next_peak(max_days: int = 60, progress=None) -> pd.Timestamp | None:
    """Advance to the next D-1 decision whose forecast recommends an event. None if none in `max_days`."""
    now = clock()
    fc = orchestrator().forecaster
    for k in range(max_days):
        day = now.normalize() + pd.Timedelta(days=k)
        decision = _daily(day - pd.Timedelta(minutes=1), settings.DECISION_TIME)
        if decision <= now or decision > DATA_END:
            continue
        ahead = fc.day_ahead(day.date())
        if ahead is not None and len(ahead.recommendations):
            return advance_to(decision, progress)
    return None


# ------------------------------------------------------------------ operator actions
def confirm(event_id: int):
    orchestrator().confirm(event_id, clock(), by="operator")


def veto(event_id: int):
    orchestrator().veto(event_id, by="operator")


def cancel(event_id: int, reason: str = "operator"):
    orchestrator().cancel(event_id, clock(), reason)


def set_level(event_id: int, level: int):
    orchestrator().set_level(event_id, level)


def status():
    from backend.intelligence.orchestrator import status as st
    return st()


# ------------------------------------------------------------------ queries
def _frame(model, *where) -> pd.DataFrame:
    with _session()() as s:
        stmt = select(model)
        for w in where:
            stmt = stmt.where(w)
        return pd.read_sql(stmt, s.bind)


def events() -> pd.DataFrame:
    return _frame(DREvent).sort_values("start_ts", ascending=False)


def targets(event_id: int | None = None, client_id: str | None = None) -> pd.DataFrame:
    where = [EventTarget.event_id == event_id] if event_id is not None else []
    if client_id is not None:
        where.append(EventTarget.client_id == client_id)
    return _frame(EventTarget, *where).sort_values(["event_id", "rank"])


def notifications(client_id: str | None = None, event_id: int | None = None, until=None) -> pd.DataFrame:
    where = []
    if client_id is not None:
        where.append(Notification.client_id == client_id)
    if event_id is not None:
        where.append(Notification.event_id == event_id)
    df = _frame(Notification, *where)
    until = until if until is not None else clock()
    if until is not None and len(df):
        df = df[pd.to_datetime(df.sent_at) <= until]
    return df.sort_values("sent_at", ascending=False)


def event_bills(client_id: str | None = None, event_id: int | None = None) -> pd.DataFrame:
    where = []
    if client_id is not None:
        where.append(EventBill.client_id == client_id)
    if event_id is not None:
        where.append(EventBill.event_id == event_id)
    return _frame(EventBill, *where)


def monthly_bills(client_id: str | None = None) -> pd.DataFrame:
    return _frame(MonthlyBill, *([MonthlyBill.client_id == client_id] if client_id else [])).sort_values("month")


def household_events(client_id: str) -> pd.DataFrame:
    """One row per event the household was informed of: window, level, answer, real kWh and surcharge."""
    ev = events()
    tg = targets(client_id=client_id)
    if tg.empty:
        return pd.DataFrame()
    df = tg.merge(ev[["event_id", "start_ts", "end_ts", "level", "status"]], on="event_id")
    df = df[df.status.isin(["notified", "active", "done", "billed"])]
    bills = event_bills(client_id=client_id)[["event_id", "window_kwh", "surcharge_dt", "missing_slots", "provisional"]]
    return df.merge(bills, on="event_id", how="left").sort_values("start_ts", ascending=False)


# ------------------------------------------------------------------ household side
def preferences(client_id: str) -> dict:
    with _session()() as s:
        p = s.get(UserPreference, client_id)
        refused = {a.device for a in s.scalars(select(AppliancePreference).where(
            AppliancePreference.client_id == client_id, AppliancePreference.allow_recommendation.is_(False)))}
        c = s.get(Client, client_id)
        return {"language": p.language, "quiet_start": p.quiet_start, "quiet_end": p.quiet_end,
                "reminders_on": p.reminders_on, "refused": refused,
                "devices": [d for d, has in [("ac", c.has_ac), ("water_heater", c.has_electric_water_heater),
                                             ("washing_machine", c.has_washing_machine)] if has]}


def save_preferences(client_id: str, language: str, quiet_start: str, quiet_end: str, reminders_on: bool,
                     refused: set[str]) -> None:
    with _session()() as s:
        p = s.get(UserPreference, client_id)
        p.language, p.quiet_start, p.quiet_end, p.reminders_on = language, quiet_start, quiet_end, reminders_on
        for dev in DEVICES:
            ap = s.get(AppliancePreference, (client_id, dev))
            if ap is None:
                ap = AppliancePreference(client_id=client_id, device=dev)
                s.add(ap)
            ap.allow_recommendation = dev not in refused
        s.commit()


def current_event(client_id: str):
    """(DREvent, EventTarget) the household has been alerted about and that has not ended yet, else (None, None)."""
    now = clock()
    with _session()() as s:
        rows = s.execute(select(DREvent, EventTarget).join(EventTarget).where(
            EventTarget.client_id == client_id, DREvent.status.in_(["notified", "active"]))).all()
    rows = [(e, t) for e, t in rows if now is not None and pd.Timestamp(e.end_ts) > now]
    return min(rows, key=lambda r: r[0].start_ts) if rows else (None, None)


def respond(event_id: int, client_id: str, response: str) -> None:
    orchestrator().record_response(event_id, client_id, response, clock())


def chat_context(client_id: str):
    from backend.orchestration.notifications.assistant import ChatContext
    ev, t = current_event(client_id)
    hist = household_events(client_id)
    last_bill = None
    if len(hist) and hist.surcharge_dt.notna().any():
        b = hist[hist.surcharge_dt.notna()].iloc[0]
        last_bill = type("Bill", (), {"event_start": b.start_ts, "window_kwh": float(b.window_kwh),
                                      "surcharge_dt": float(b.surcharge_dt)})()
    comparison = None
    try:
        from backend.services import segmentation_service as seg
        if seg.profiling_available():
            row = seg.household_segments().set_index("client_id").loc[client_id]
            comparison = row.get("comparison_message")
    except (ImportError, KeyError):
        pass
    return ChatContext(now=clock(), lang=preferences(client_id)["language"], event=ev, target=t, last_bill=last_bill,
                       n_events=len(hist), n_accepted=int((hist.response == "accept").sum()) if len(hist) else 0,
                       total_surcharge=float(hist.surcharge_dt.fillna(0).sum()) if len(hist) else 0.0,
                       comparison_message=comparison)


def chat(client_id: str, text: str) -> tuple[str, str]:
    from backend.orchestration.notifications.assistant import reply
    ctx = chat_context(client_id)
    rec = (lambda r: respond(ctx.event.event_id, client_id, r)) if ctx.event is not None else None
    return reply(text, ctx, rec)
