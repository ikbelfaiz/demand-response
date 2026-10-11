"""Replays the DR orchestrator on dataset v3 with a simulated clock, using the team's community
forecast model and decision policy (every job runs exactly as in the app).

    python scripts/run_orchestrator_replay.py --start 2025-06-01 --end 2025-09-30

Household responses are simulated from each household's acceptance rate (data/customer_profiles.csv).
Outputs in outputs/orchestrator/: events.csv, targets.csv, notifications.csv, event_bills.csv,
monthly_bills.csv, report.txt
"""
import argparse

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.intelligence import pricing
from backend.orchestration import data_access, settings as config
from backend.orchestration.lifecycle import Orchestrator
from backend.orchestration.notifications.render import render_for_target
from backend.orchestration.store import (DREvent, EventBill, EventTarget, MonthlyBill, Notification,
                                         get_sessionmaker, init_db)


def make_response_simulator(seed: int = 11):
    rng = np.random.default_rng(seed)

    def respond(target, household, first: bool) -> str:
        u = rng.random()
        if first and u < 0.08:
            return "later"
        if u < float(household["acceptance_rate"]):
            return "accept"
        return "decline" if rng.random() < 0.6 else "no_response"
    return respond


def run(start: str, end: str, db_url: str | None = None, quiet: bool = False, forecaster=None) -> dict:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    url = db_url or f"sqlite:///{config.OUT_DIR / 'replay.db'}"
    engine = init_db(url)
    Session = get_sessionmaker(engine)
    orch = Orchestrator(Session, forecaster, auto_confirm=True, response_simulator=make_response_simulator(),
                        log=(lambda *_: None) if quiet else print)
    grid30 = data_access.grid30()
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=4)   # +4 days: last bills and month close
    for now in pd.date_range(t0, t1, freq=config.SLOT):
        orch.tick(now)
    return report(Session, grid30, start, end)


def report(Session, grid30, start, end) -> dict:
    out = config.OUT_DIR
    grid30 = grid30.reset_index().set_index("timestamp")
    with Session() as s:
        dfs = {name: pd.read_sql(select(model), s.bind) for name, model in
               [("events", DREvent), ("targets", EventTarget), ("notifications", Notification),
                ("event_bills", EventBill), ("monthly_bills", MonthlyBill)]}
    for name, df in dfs.items():
        df.to_csv(out / f"{name}.csv", index=False)
    ev, tg, nt, eb, mb = (dfs[k] for k in ["events", "targets", "notifications", "event_bills", "monthly_bills"])

    period = grid30.loc[start:pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=30)]
    peak_days = set(period.index[period["is_dr_peak"] == 1].date)
    kept = ev[ev.status.isin(["billed", "done", "active", "notified"])]
    ev_days = set(pd.to_datetime(kept.start_ts).dt.date)
    nt["week"] = pd.to_datetime(nt.sent_at).dt.isocalendar().week
    alerts = nt[nt.kind.str.startswith("alert")]
    per_week = alerts.groupby(["client_id", "week"]).event_id.nunique()

    lines = [
        f"Replay {start} → {end}",
        f"Events: proposed {len(ev)}, cancelled {int((ev.status == 'cancelled').sum())}, billed {int((ev.status == 'billed').sum())}",
        f"Forecast model: {ev.forecast_model_version.iloc[0] if len(ev) else '-'} | decision policy: "
        f"{ev.decision_policy_version.iloc[0] if len(ev) else '-'}",
        f"Peak days in dataset: {len(peak_days)} | events on peak days: {len(ev_days & peak_days)} | "
        f"false alarms: {len(ev_days - peak_days)} | missed peak days: {len(peak_days - ev_days)}",
        f"Level mix: {ev.level.value_counts().sort_index().to_dict()}",
        f"Notifications by kind: {nt.kind.value_counts().to_dict()}",
        f"Distinct events alerted per household per week: max {int(per_week.max()) if len(per_week) else 0}",
        f"Responses: {tg.response.value_counts().to_dict()}",
        f"Personalised alerts: {int(tg.personalised.sum())} of {len(tg)} | devices: "
        f"{tg.recommended_device.value_counts().to_dict()}",
        f"Event surcharges billed (final): {eb.surcharge_dt.sum():.3f} DT over {len(eb)} household-events",
        f"Monthly bills: {len(mb)} | cap applied: {int(mb.cap_applied.sum())} | "
        f"mean DR surcharge per household-month: {mb.dr_surcharge_dt.mean():.3f} DT",
    ]
    # sample messages: the first personalised target, in the three languages
    ex = tg[tg.personalised & ~tg.exempt].head(1)
    if len(ex):
        t = ex.iloc[0]
        e = ev.set_index("event_id").loc[t.event_id]
        e_ns = type("E", (), {"start_ts": pd.Timestamp(e.start_ts), "end_ts": pd.Timestamp(e.end_ts)})
        lines.append(f"\nSample alert — event {t.event_id}, household {t.client_id} ({t.segment}):")
        for lang in ["en", "fr", "ar"]:
            text, _ = render_for_target("alert_personal", lang, e_ns, t,
                                        level_pct=round(pricing.LEVELS[int(e.level)] * 100))
            lines += [f"[{lang}]", text]
    for kind in ["reminder", "summary", "cancellation"]:
        sample = nt[nt.kind == kind].head(1)
        if len(sample):
            lines += [f"\nSample {kind} ({sample.iloc[0].language}, {sample.iloc[0].client_id}):", sample.iloc[0].text]
    txt = "\n".join(lines)
    (out / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return {"events": ev, "targets": tg, "notifications": nt, "event_bills": eb, "monthly_bills": mb,
            "peak_days": peak_days, "event_days": ev_days}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2025-09-30")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    run(a.start, a.end, a.db)


if __name__ == "__main__":
    main()
