"""DR event lifecycle. `Orchestrator.tick(now)` runs every job due at `now` (every 30 min).

    D-1 14:30  daily_decision      team's community forecast (issued 14:00) → recommended events → `proposed`
    operator   confirm / veto      customer profiling priority list + NILM device recommendation + pricing
                                   → one EventTarget per household
    D-1 18:00  send_alerts         alerts in each household's language, quiet hours respected
    D-1 21:00  send_later          second alert for households that answered "Later"
    operator   cancel              cancellation message, no surcharge
    start      activate            status `active`
    start+30   check_first_slot    one reminder if the first slot is above the household's usual pace
    end        close               status `done`
    D+1 09:00  bill_and_summarise  real surcharge (eq. 7) from the meter, summary message, `billed`
    3rd 06:00  close_month         monthly bills with the real band and the 10% cap
The system never acts on appliances: households decide and act by hand.
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.intelligence import pricing
from backend.orchestration import data_access, settings as config
from backend.orchestration.adapters.devices import default_recommender
from backend.orchestration.adapters.profiling import default_targeting, month_projection
from backend.orchestration.decide import decide
from backend.orchestration.notifications.render import render_for_target
from backend.orchestration.plan import build_targets
from backend.orchestration.store import (AppliancePreference, Client, DREvent, EventBill, EventTarget, Forecast,
                                         MonthlyBill, Notification, UserPreference)

ACTIVE_RESPONSES = ("accept", "no_response", "later")


INACTIVE = ["vetoed", "cancelled", "expired"]   # events that never reach households

def _in_quiet(t: pd.Timestamp, start: str, end: str) -> bool:
    hm = t.strftime("%H:%M")
    return (start <= hm or hm < end) if start > end else (start <= hm < end)


class Orchestrator:
    def __init__(self, Session, forecaster=None, recommender=None, targeting=None, *, grid30: pd.DataFrame | None = None,
                 meter30: pd.DataFrame | None = None, households: pd.DataFrame | None = None,
                 auto_confirm: bool = False, response_simulator: Callable | None = None, log: Callable = print):
        self.Session = Session
        if forecaster is None:
            from backend.orchestration.adapters.forecast import CommunityDRForecaster
            forecaster = CommunityDRForecaster()
        self.forecaster = forecaster
        self.recommender = recommender or default_recommender()      # NILM model, else declared-device rules
        self.targeting = targeting or default_targeting()            # customer profiling model, else local list
        self.grid30 = data_access.grid30() if grid30 is None else grid30
        self.meter30 = data_access.meter30() if meter30 is None else meter30
        self.households = data_access.households() if households is None else households
        self.auto_confirm = auto_confirm
        self.respond = response_simulator          # replay only; in the app responses come from the UI
        self.log = log

    # ------------------------------------------------------------------ dispatcher
    def tick(self, now: pd.Timestamp):
        hm = now.strftime("%H:%M")
        if hm == config.DECISION_TIME:
            self.daily_decision(now)
        if hm == config.ALERT_TIME:
            self.send_alerts(now)
        if hm == config.LATER_TIME:
            self.send_later(now)
        self.run_event_clock(now)
        if hm == config.BILLING_TIME:
            self.bill_and_summarise(now)
        if now.day == config.MONTH_CLOSE_DAY and hm == config.MONTH_CLOSE_TIME:
            self.close_month(now)

    # ------------------------------------------------------------------ helpers
    def _event_days(self, s) -> set[date]:
        rows = s.scalars(select(DREvent).where(DREvent.status.notin_(INACTIVE))).all()
        return {e.start_ts.date() for e in rows}

    def _prefs(self, s) -> dict[str, UserPreference]:
        return {p.client_id: p for p in s.scalars(select(UserPreference)).all()}

    def _notify(self, s, now, target, event, kind, prefs, **kw):
        pref = prefs[target.client_id]
        sent = now
        if _in_quiet(now, pref.quiet_start, pref.quiet_end):          # wait for the end of quiet hours
            h, m = map(int, pref.quiet_end.split(":"))
            sent = (now + pd.Timedelta(days=1 if now.hour >= h else 0)).normalize() + pd.Timedelta(hours=h, minutes=m)
        text, payload = render_for_target(kind, pref.language, event, target,
                                          level_pct=round(pricing.LEVELS[event.level] * 100), **kw)
        s.add(Notification(client_id=target.client_id, event_id=event.event_id, kind=kind, language=pref.language,
                           channel=pref.channel, text=text, payload=payload, sent_at=sent.to_pydatetime()))

    @staticmethod
    def _alert_kind(t: EventTarget) -> str:
        return "alert_exempt" if t.exempt else ("alert_personal" if t.personalised else "alert_general")

    # ------------------------------------------------------------------ D-1 14:30
    def daily_decision(self, now: pd.Timestamp):
        day_ahead = self.forecaster.day_ahead(now.date())
        if day_ahead is None:                     # the community model cannot forecast this day
            return None
        fc = day_ahead.forecast
        with self.Session() as s:
            s.add_all([Forecast(issued_at=day_ahead.issue_time.to_pydatetime(), target_ts=r.timestamp.to_pydatetime(),
                                predicted_demand_kw=float(r.predicted_demand_kw),
                                predicted_supply_kw=float(r.predicted_supply_kw),
                                predicted_margin_kw=float(r.predicted_margin_kw), model_version=day_ahead.model_version)
                       for r in fc.itertuples()])
            target = day_ahead.target_day
            week_start = target - pd.Timedelta(days=target.dayofweek)
            n_week = len(s.scalars(select(DREvent).where(
                DREvent.status.notin_(INACTIVE),
                DREvent.start_ts >= week_start.to_pydatetime(),
                DREvent.start_ts < (week_start + pd.Timedelta(days=7)).to_pydatetime())).all())
            d = decide(fc, day_ahead.recommendations, n_week)
            if d is None:
                s.commit()
                return None
            ev = DREvent(start_ts=d.start.to_pydatetime(), end_ts=d.end.to_pydatetime(), level=d.level,
                         status="proposed", max_deficit_kw=d.max_deficit_kw, shortage_energy_kwh=d.shortage_energy_kwh,
                         deficit_share=d.deficit_share, recommended_reduction_pct=d.recommended_reduction_pct,
                         forecast_model_version=day_ahead.model_version,
                         decision_policy_version=day_ahead.policy_version, created_at=now.to_pydatetime())
            s.add(ev)
            s.commit()
            event_id = ev.event_id
        self.log(f"{now:%Y-%m-%d %H:%M} proposed event {event_id}: {d.start:%Y-%m-%d %H:%M}–{d.end:%H:%M}, "
                 f"level {d.level}, max deficit {d.max_deficit_kw:.1f} kW, shortage {d.deficit_share:.1%} of supply")
        if self.auto_confirm:
            self.confirm(event_id, now, by="auto")
        return event_id

    @staticmethod
    def confirm_deadline(start_ts) -> pd.Timestamp:
        """Last moment to confirm an event: the alerts of the day before."""
        h, m = map(int, config.ALERT_TIME.split(":"))
        return pd.Timestamp(start_ts).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=h, minutes=m)

    # ------------------------------------------------------------------ operator actions
    def confirm(self, event_id: int, now: pd.Timestamp, by: str = "operator"):
        with self.Session() as s:
            ev = s.get(DREvent, event_id)
            if ev.status != "proposed" or now >= self.confirm_deadline(ev.start_ts):
                return
            start, end = pd.Timestamp(ev.start_ts), pd.Timestamp(ev.end_ts)
            prio = self.targeting.priority_list(self.meter30, self.households, start, end, now,
                                                self._event_days(s) - {start.date()}, ev.shortage_energy_kwh)
            refused = {}
            for ap in s.scalars(select(AppliancePreference).where(AppliancePreference.allow_recommendation.is_(False))):
                refused.setdefault(ap.client_id, set()).add(ap.device)
            # temperature known at decision time: same window two days earlier (no target-day weather is used)
            lag = pd.Timedelta(days=2)
            mean_temp = float(self.grid30["temperature_c"].loc[start - lag:end - lag - pd.Timedelta(seconds=1)].mean())
            s.add_all(build_targets(ev, prio, self.households, refused, mean_temp, self.recommender, asof=now))
            ev.status, ev.confirmed_by = "confirmed", by
            ev.profiling_version = getattr(self.targeting, "source", None)
            ev.device_model = getattr(self.recommender, "source", None)
            s.commit()

    def set_level(self, event_id: int, level: int, by: str = "operator"):
        """The operator can change the surcharge level of a proposed event before confirming it."""
        if level not in pricing.LEVELS:
            raise ValueError(f"Unknown surcharge level {level}")
        with self.Session() as s:
            ev = s.get(DREvent, event_id)
            if ev.status == "proposed":
                ev.level = level
                s.commit()

    def veto(self, event_id: int, by: str = "operator"):
        with self.Session() as s:
            ev = s.get(DREvent, event_id)
            if ev.status == "proposed":
                ev.status, ev.confirmed_by = "vetoed", by
                s.commit()

    def cancel(self, event_id: int, now: pd.Timestamp, reason: str = "operator"):
        """Cancel a confirmed or notified event before it starts: everyone is told, no surcharge."""
        with self.Session() as s:
            ev = s.get(DREvent, event_id)
            if ev.status not in ("confirmed", "notified") or now >= pd.Timestamp(ev.start_ts):
                return
            prefs = self._prefs(s)
            if ev.status == "notified":
                for t in ev.targets:
                    self._notify(s, now, t, ev, "cancellation", prefs)
            ev.status, ev.cancel_reason = "cancelled", reason
            s.commit()

    def record_response(self, event_id: int, client_id: str, response: str, now: pd.Timestamp):
        """Called by the consumer page when a household taps Accept / Decline / Later."""
        with self.Session() as s:
            t = s.scalars(select(EventTarget).where(EventTarget.event_id == event_id,
                                                    EventTarget.client_id == client_id)).one()
            t.response, t.response_at = response, now.to_pydatetime()
            s.commit()

    # ------------------------------------------------------------------ D-1 18:00 and 21:00
    def send_alerts(self, now: pd.Timestamp):
        tomorrow = now.normalize() + pd.Timedelta(days=1)
        with self.Session() as s:
            for ev in s.scalars(select(DREvent).where(DREvent.status == "proposed")).all():
                if pd.Timestamp(ev.start_ts).normalize() == tomorrow:     # not confirmed before the alert time
                    ev.status = "expired"
                    self.log(f"{now:%Y-%m-%d %H:%M} event {ev.event_id} expired (not confirmed before the alerts)")
            prefs = self._prefs(s)
            evs = s.scalars(select(DREvent).where(DREvent.status == "confirmed")).all()
            for ev in [e for e in evs if pd.Timestamp(e.start_ts).normalize() == tomorrow]:
                for t in sorted(ev.targets, key=lambda x: x.rank):
                    self._notify(s, now, t, ev, self._alert_kind(t), prefs)
                    if self.respond:
                        t.response = self.respond(t, self.households.loc[t.client_id], first=True)
                        t.response_at = now.to_pydatetime()
                ev.status = "notified"
            s.commit()

    def send_later(self, now: pd.Timestamp):
        tomorrow = now.normalize() + pd.Timedelta(days=1)
        with self.Session() as s:
            prefs = self._prefs(s)
            for ev in s.scalars(select(DREvent).where(DREvent.status == "notified")).all():
                if pd.Timestamp(ev.start_ts).normalize() != tomorrow:
                    continue
                for t in [t for t in ev.targets if t.response == "later"]:
                    self._notify(s, now, t, ev, self._alert_kind(t), prefs)
                    if self.respond:
                        t.response = self.respond(t, self.households.loc[t.client_id], first=False)
                        t.response_at = now.to_pydatetime()
            s.commit()

    # ------------------------------------------------------------------ during the event
    def run_event_clock(self, now: pd.Timestamp):
        with self.Session() as s:
            for ev in s.scalars(select(DREvent).where(DREvent.status == "proposed")).all():
                if now >= pd.Timestamp(ev.start_ts):         # never confirmed by the operator
                    ev.status = "expired"
            evs = s.scalars(select(DREvent).where(DREvent.status.in_(["notified", "active"]))).all()
            prefs = None
            for ev in evs:
                start, end = pd.Timestamp(ev.start_ts), pd.Timestamp(ev.end_ts)
                if ev.status == "notified" and now == start:
                    ev.status = "active"
                if ev.status == "active" and now == start + pd.Timedelta(config.SLOT):
                    prefs = prefs or self._prefs(s)
                    self.check_first_slot(s, ev, now, prefs)
                if ev.status == "active" and now == end:
                    ev.status = "done"
            s.commit()

    def check_first_slot(self, s, ev: DREvent, now: pd.Timestamp, prefs):
        start, end = pd.Timestamp(ev.start_ts), pd.Timestamp(ev.end_ts)
        n_slots = int((end - start) / pd.Timedelta(config.SLOT))
        first = self.meter30.loc[start]
        for t in ev.targets:
            if t.exempt or t.response not in ACTIVE_RESPONSES or not prefs[t.client_id].reminders_on:
                continue
            kwh = first.get(t.client_id, np.nan)
            pace = t.usual_window_kwh / n_slots
            if np.isnan(kwh) or pace <= 0 or kwh <= pace:
                continue
            band = pricing.Band(t.band_price, t.exempt)
            proj = pricing.projected_extra_cost(float(kwh), pace * (n_slots - 1), band, ev.level)
            self._notify(s, now, t, ev, "reminder", prefs, projected=proj)

    # ------------------------------------------------------------------ D+1 09:00
    def _month_band(self, client_id: str, kva: float, month_start: pd.Timestamp, now: pd.Timestamp):
        next_month = month_start + pd.offsets.MonthBegin(1)
        asof = min(now, next_month - pd.Timedelta(seconds=1))
        _, proj = month_projection(self.meter30[[client_id]], asof)
        return pricing.band_price(float(proj.iloc[0]), kva)

    def bill_and_summarise(self, now: pd.Timestamp):
        day = now.normalize() - pd.Timedelta(days=1)            # events of yesterday
        with self.Session() as s:
            prefs = self._prefs(s)
            kva = {c.client_id: c.contracted_kva for c in s.scalars(select(Client)).all()}
            for ev in s.scalars(select(DREvent).where(DREvent.status == "done")).all():
                start, end = pd.Timestamp(ev.start_ts), pd.Timestamp(ev.end_ts)
                if start.normalize() != day:
                    continue
                window = self.meter30.loc[start:end - pd.Timedelta(seconds=1)]
                for t in ev.targets:
                    col = window[t.client_id]
                    kwh, missing = float(col.sum(skipna=True)), int(col.isna().sum())   # missing slots never billed
                    band = self._month_band(t.client_id, kva[t.client_id], start.replace(day=1), now)
                    bill = EventBill(event_id=ev.event_id, client_id=t.client_id, window_kwh=round(kwh, 3),
                                     missing_slots=missing, band_price=band.price, level=ev.level,
                                     window_cost_dt=pricing.window_cost(kwh, band, ev.level),
                                     surcharge_dt=pricing.event_surcharge(kwh, band, ev.level), provisional=True)
                    s.add(bill)
                    if not band.economic:                      # economic band: nothing billed, no summary
                        view = SimpleNamespace(client_id=t.client_id, peak_price=pricing.peak_price(band, ev.level),
                                               band_price=band.price, action=t.action,
                                               expected_saving_dt=0.0, expected_surcharge_dt=0.0)
                        self._notify(s, now, view, ev, "summary", prefs, bill=bill)
                ev.status = "billed"
            s.commit()

    # ------------------------------------------------------------------ monthly close
    def close_month(self, now: pd.Timestamp):
        month_start = (now.normalize().replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        month_end = month_start + pd.offsets.MonthBegin(1)
        tag = month_start.strftime("%Y-%m")
        totals = self.meter30.loc[month_start:month_end - pd.Timedelta(seconds=1)].sum()
        with self.Session() as s:
            events = {e.event_id for e in s.scalars(select(DREvent).where(DREvent.status == "billed")).all()
                      if month_start <= pd.Timestamp(e.start_ts) < month_end}
            bills = s.scalars(select(EventBill).where(EventBill.event_id.in_(list(events) or [-1]))).all()
            by_client = {}
            for b in bills:
                by_client.setdefault(b.client_id, []).append(b)
            for c in s.scalars(select(Client)).all():
                M = float(totals.get(c.client_id, 0.0))
                ebs = by_client.get(c.client_id, [])
                res = pricing.monthly_bill(M, c.contracted_kva, [{"window_kwh": b.window_kwh, "level": b.level} for b in ebs])
                band = pricing.band_price(M, c.contracted_kva)
                for b in ebs:                                    # final event amounts with the real band
                    b.band_price = band.price
                    b.surcharge_dt = pricing.event_surcharge(b.window_kwh, band, b.level)
                    b.window_cost_dt = pricing.window_cost(b.window_kwh, band, b.level)
                    b.provisional = False
                s.add(MonthlyBill(client_id=c.client_id, month=tag, monthly_kwh=round(M, 1),
                                  band_price=res["band_price"], energy_dt=res["energy_dt"],
                                  dr_surcharge_raw_dt=res["dr_surcharge_raw_dt"], dr_surcharge_dt=res["dr_surcharge_dt"],
                                  cap_applied=res["cap_applied"], power_fee_dt=res["power_fee_dt"],
                                  total_before_tax_dt=res["total_before_tax_dt"]))
            s.commit()
        self.log(f"{now:%Y-%m-%d %H:%M} monthly bills closed for {tag}")
