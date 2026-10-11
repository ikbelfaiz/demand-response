"""DR orchestrator pages: My DR Assistant (consumer), orchestrator events in My DR Participation (consumer),
and DR Orchestrator (operator). All state comes from backend.services.orchestrator_service (demo clock)."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.intelligence import pricing
from backend.services import orchestrator_service as svc
from frontend.charts import style
from frontend.components import page_header, render_chart

CSS = """
<style>
.dr-clock{background:#0F172A;color:#E2E8F0;border-radius:12px;padding:.6rem 1rem;margin:.4rem 0 1rem;font-size:.92rem}
.dr-clock b{color:#FFFFFF}
.dr-card{background:#FFFFFF;border:1px solid #E2E8F0;border-left:6px solid #D97706;border-radius:14px;
         padding:1rem 1.2rem;margin:.4rem 0 .8rem;box-shadow:0 5px 18px rgba(15,23,42,.06);color:#1E293B}
.dr-card.exempt{border-left-color:#0F766E}
.dr-card .t{font-size:1.05rem;font-weight:700;margin-bottom:.4rem;color:#0F172A}
.dr-card .row{margin:.18rem 0;line-height:1.5}
.dr-card .muted{color:#64748B;font-size:.85rem}
.dr-pill{display:inline-block;border-radius:999px;padding:.12rem .6rem;font-size:.78rem;font-weight:700;margin-right:.35rem}
.dr-pill.on{background:#DCFCE7;color:#166534}.dr-pill.off{background:#F1F5F9;color:#475569}
.dr-pill.warn{background:#FEF3C7;color:#92400E}
[dir=rtl]{text-align:right}
</style>
"""
STATUS_ORDER = ["proposed", "confirmed", "notified", "active", "done", "billed", "vetoed", "cancelled", "expired"]
RESPONSE_COLORS = {"accept": "#16A34A", "later": "#2563EB", "decline": "#DC2626", "no_response": "#94A3B8"}
KIND_LABEL = {"alert_personal": "Alert", "alert_general": "Alert", "alert_exempt": "Information",
              "reminder": "Reminder", "summary": "Summary", "cancellation": "Cancellation"}


def _css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _clock_banner(operator: bool = False) -> pd.Timestamp | None:
    now = svc.clock()
    if now is None:
        return None
    nxt, label = svc.next_milestone(now)
    who = "You control the clock below." if operator else "The operator moves this clock on the DR Orchestrator page."
    st.markdown(f'<div class="dr-clock">🕒 Demo time <b>{now:%A %d %B %Y, %H:%M}</b> · next: {nxt:%d/%m %H:%M} '
                f'— {html.escape(label)} · {who}</div>', unsafe_allow_html=True)
    return now


def _not_ready(operator: bool) -> None:
    if operator:
        st.info("The orchestrator demo has not been started yet. Build the history below to create past events.")
    else:
        st.info("The DR orchestrator demo has not been started yet. Open Operator Dashboard ▸ DR Orchestrator "
                "and click “Build history”.")


def _text_html(text: str, lang: str) -> str:
    body = html.escape(text).replace("\n", "<br>")
    return f'<div dir="rtl">{body}</div>' if lang == "ar" else body


# ====================================================================== consumer: My DR Assistant
def assistant(_frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    _css()
    page_header("Consumer / chatbot", "My DR Assistant",
                "Peak alerts, your answer, and questions about the peak price. You act by hand: nothing is switched off for you.")
    if not svc.initialised():
        _not_ready(False)
        return
    now = _clock_banner()
    prefs = svc.preferences(client_id)
    lang = prefs["language"]

    ev, t = svc.current_event(client_id)
    if ev is None:
        st.markdown('<div class="dr-card exempt"><div class="t">No peak planned</div>'
                    '<div class="row">You will get an alert the day before if the forecast shows a peak.</div></div>',
                    unsafe_allow_html=True)
    else:
        _event_card(ev, t, client_id, now)

    st.subheader("Messages")
    _chat(client_id, lang)

    with st.expander("My preferences"):
        _preferences_form(client_id, prefs)


def _event_card(ev, t, client_id: str, now: pd.Timestamp) -> None:
    started = now >= pd.Timestamp(ev.start_ts)
    pct = round(pricing.LEVELS[ev.level] * 100)
    day = "Today" if pd.Timestamp(ev.start_ts).normalize() == now.normalize() else "Tomorrow"
    window = f"{pd.Timestamp(ev.start_ts):%H:%M} – {pd.Timestamp(ev.end_ts):%H:%M}"
    state = "active now" if started else day.lower()
    if t.exempt:
        rows = [f"Peak window: <b>{window}</b> ({state})",
                f"Your price does not change: <b>{t.band_price} millimes/kWh</b> (economic band).",
                f"You can still help: {html.escape(_action(ev, t))}."]
        klass = "exempt"
    else:
        rows = [f"Peak window: <b>{window}</b> ({state})",
                f"Peak price: <b>{t.peak_price} millimes/kWh</b> (+{pct}% vs STEG price of {t.band_price} millimes)",
                f"Recommended action: <b>{html.escape(_action(ev, t))}</b>"]
        if t.personalised and t.expected_saving_dt > 0:
            rows.append(f"Expected saving: <b>~{t.expected_saving_dt:.2f} DT</b> if you reduce")
        rows.append(f'<span class="muted">If you use as much as usual ({t.usual_window_kwh:.2f} kWh), '
                    f'extra cost ≈ {t.expected_surcharge_dt:.2f} DT.</span>')
        klass = ""
    answer = {"accept": "✅ You accepted", "decline": "❌ You declined", "later": "⏰ You asked for a reminder",
              "no_response": "No answer yet"}[t.response]
    st.markdown(f'<div class="dr-card {klass}"><div class="t">⚡ Critical peak pricing — level {ev.level}</div>'
                + "".join(f'<div class="row">{r}</div>' for r in rows)
                + f'<div class="row muted">{answer}</div></div>', unsafe_allow_html=True)
    a, b, c, _ = st.columns([1, 1, 1, 3])
    for col, resp, label in [(a, "accept", "Accept"), (b, "decline", "Decline"), (c, "later", "Later")]:
        if col.button(label, key=f"resp_{resp}_{ev.event_id}_{client_id}", disabled=started or t.response == resp,
                      type="primary" if resp == "accept" else "secondary", width="stretch"):
            svc.respond(ev.event_id, client_id, resp)
            st.rerun()
    if started:
        st.caption("The peak has started: your answer can no longer be changed.")


def _action(ev, t) -> str:
    from backend.orchestration.notifications.render import action_text
    return action_text(t.action, "en", f"{pd.Timestamp(ev.end_ts):%H:%M}")


def _chat(client_id: str, lang: str) -> None:
    key = f"dr_chat_{client_id}"
    history = st.session_state.setdefault(key, [])
    inbox = svc.notifications(client_id).sort_values("sent_at").tail(12)
    box = st.container(height=430)
    with box:
        if inbox.empty and not history:
            st.caption("No message yet.")
        for n in inbox.itertuples():
            with st.chat_message("assistant", avatar="⚡"):
                st.caption(f"{KIND_LABEL.get(n.kind, n.kind)} · {pd.Timestamp(n.sent_at):%d/%m %H:%M}")
                st.markdown(_text_html(n.text, n.language), unsafe_allow_html=True)
        for role, text in history:
            with st.chat_message(role, avatar="⚡" if role == "assistant" else None):
                st.markdown(_text_html(text, lang) if role == "assistant" else html.escape(text),
                            unsafe_allow_html=True)
    prompt = st.chat_input({"fr": "Posez une question (prix, quand, que faire…)",
                            "ar": "اطرح سؤالًا (السعر، متى، ماذا أفعل…)"}.get(lang, "Ask a question (price, when, what to do…)"),
                           key=f"dr_chat_input_{client_id}")
    if prompt:
        _, text = svc.chat(client_id, prompt)
        history += [("user", prompt), ("assistant", text)]
        st.rerun()
    st.caption("Answers use the prices and estimates computed by the pricing engine; no number is generated freely.")


def _preferences_form(client_id: str, prefs: dict) -> None:
    with st.form(f"prefs_{client_id}"):
        langs = list(svc.LANGUAGES)
        language = st.selectbox("Language", langs, index=langs.index(prefs["language"]),
                                format_func=svc.LANGUAGES.get)
        a, b = st.columns(2)
        quiet_start = a.text_input("No messages from", prefs["quiet_start"])
        quiet_end = b.text_input("until", prefs["quiet_end"])
        reminders = st.checkbox("Reminder during the peak if my consumption stays high", prefs["reminders_on"])
        devices = prefs["devices"]
        allowed = st.multiselect("Appliances the assistant may recommend", devices,
                                 default=[d for d in devices if d not in prefs["refused"]],
                                 format_func=svc.DEVICES.get)
        if st.form_submit_button("Save"):
            svc.save_preferences(client_id, language, quiet_start, quiet_end, reminders,
                                 {d for d in svc.DEVICES if d not in allowed})
            st.success("Saved. New messages will use these preferences.")


# ====================================================================== consumer: My DR Participation (section)
def participation_section(client_id: str) -> None:
    """Real answers and extra cost of each critical peak pricing event (replaces the demo radio)."""
    _css()
    st.subheader("Critical peak pricing events (DR orchestrator)")
    if not svc.initialised():
        _not_ready(False)
        return
    _clock_banner()
    hist = svc.household_events(client_id)
    if hist.empty:
        st.caption("You have not been informed of any peak event yet.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Peak events", len(hist))
    b.metric("Accepted", int((hist.response == "accept").sum()))
    c.metric("Energy in peak windows", f"{hist.window_kwh.sum():.2f} kWh")
    d.metric("Extra cost billed", f"{hist.surcharge_dt.fillna(0).sum():.2f} DT")
    view = pd.DataFrame({
        "date": pd.to_datetime(hist.start_ts).dt.strftime("%d/%m/%Y"),
        "window": pd.to_datetime(hist.start_ts).dt.strftime("%H:%M") + " – " + pd.to_datetime(hist.end_ts).dt.strftime("%H:%M"),
        "level": hist.level.map(lambda l: f"{l} (+{round(pricing.LEVELS[l] * 100)}%)"),
        "your answer": hist.response.str.replace("_", " "),
        "recommended": hist.action.fillna("general advice").str.replace("_", " "),
        "kWh in window": hist.window_kwh.round(2),
        "peak price (mill/kWh)": hist.peak_price,
        "extra cost (DT)": hist.surcharge_dt.round(3),
        "bill": hist.provisional.map({True: "provisional", False: "final"}).fillna("pending"),
    })
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption("Extra cost = kWh used in the window × STEG band price × surcharge (equation 7). Provisional until the "
               "monthly bill fixes the real band. Missing half-hours are never billed. Exempt (economic band): 0 DT.")
    mb = svc.monthly_bills(client_id)
    if len(mb):
        with st.expander("Monthly bills"):
            st.dataframe(mb[["month", "monthly_kwh", "band_price", "energy_dt", "dr_surcharge_dt", "cap_applied",
                             "power_fee_dt", "total_before_tax_dt"]].rename(columns={
                "monthly_kwh": "kWh", "band_price": "band (mill)", "energy_dt": "energy (DT)",
                "dr_surcharge_dt": "DR surcharges (DT)", "cap_applied": "10% cap", "power_fee_dt": "power fee (DT)",
                "total_before_tax_dt": "total before tax (DT)"}), hide_index=True, width="stretch")


# ====================================================================== operator: DR Orchestrator
def operator_page(_grid: pd.DataFrame) -> None:
    _css()
    page_header("Operator / orchestration", "DR Orchestrator",
                "Turns the D+1 forecast into critical peak pricing events: confirm or veto, follow the alerts and answers, "
                "and check the surcharges billed.")
    _model_status()
    if not svc.initialised():
        _not_ready(True)
        _history_form(first=True)
        return
    now = _clock_banner(operator=True)
    _clock_controls(now)

    ev = svc.events()
    ev["start_ts"], ev["end_ts"] = pd.to_datetime(ev.start_ts), pd.to_datetime(ev.end_ts)
    from backend.orchestration.lifecycle import Orchestrator
    _proposals(ev[(ev.status == "proposed") & (ev.start_ts.map(Orchestrator.confirm_deadline) > now)])
    _upcoming(ev[ev.status.isin(["confirmed", "notified"]) & (ev.start_ts > now)])
    _event_explorer(ev)
    with st.expander("Orchestrator log"):
        st.code("\n".join(svc.LOG) or "No log yet in this session.", language=None)


def _model_status() -> None:
    s = svc.status()
    items = [("Forecast D+1", s.decision_model, "community model"),
             ("Customer profiling", s.profiling_model, "K-means" if s.profiling_model else "local list"),
             ("NILM", s.nilm_model, "TCN model" if s.nilm_model else "declared-appliance rules"),
             ("Pricing", s.pricing_engine, "critical peak pricing"),
             ("Notifications", s.notification_engine, "FR / AR / EN")]
    st.markdown(" ".join(f'<span class="dr-pill {"on" if ok else "warn"}">{name}: {detail}</span>'
                         for name, ok, detail in items), unsafe_allow_html=True)


def _progress(label: str):
    bar = st.progress(0.0, text=label)
    return lambda x: bar.progress(min(float(x), 1.0), text=label)


def _clock_controls(now: pd.Timestamp) -> None:
    a, b, c, d = st.columns(4)
    if a.button("▶ Next step", width="stretch", help="Run the orchestrator up to its next action"):
        t, label = svc.next_step(_progress("Running…"))
        st.toast(f"{t:%d/%m %H:%M} — {label}")
        st.rerun()
    if b.button("⏭ Next forecast peak", width="stretch", help="Advance to the next D-1 decision with a forecast peak"):
        t = svc.jump_to_next_peak(progress=_progress("Advancing…"))
        st.toast(f"Decision of {t:%d/%m %H:%M}" if t is not None else "No forecast peak in the next 60 days")
        st.rerun()
    if c.button("+1 day", width="stretch"):
        svc.advance_to(now + pd.Timedelta(days=1), _progress("Advancing one day…"))
        st.rerun()
    with d.popover("Rebuild history", width="stretch"):
        _history_form(first=False)


def _history_form(first: bool) -> None:
    until = st.date_input("Replay history up to", svc.DEFAULT_HISTORY_END.date(),
                          min_value=(svc.HISTORY_START + pd.Timedelta(days=1)).date(), max_value=pd.Timestamp("2025-12-30").date(),
                          key="dr_history_until")
    st.caption(f"Replays the orchestrator from {svc.HISTORY_START:%d/%m/%Y} with simulated household answers "
               "(acceptance rate of each household). After that date, answers come from the consumer pages.")
    if st.button("Build history" if first else "Reset and rebuild", type="primary", key="dr_build_history"):
        with st.spinner("Replaying the orchestrator…"):
            svc.build_history(pd.Timestamp(until), progress=_progress("Replaying…"))
        st.rerun()


def _proposals(prop: pd.DataFrame) -> None:
    st.subheader("Proposed events")
    if prop.empty:
        st.caption("No event waiting for a decision. Proposals appear at 14:30 the day before a forecast peak and "
                   "must be confirmed before the alerts at 18:00.")
        return
    for e in prop.itertuples():
        with st.container(border=True):
            from backend.orchestration.lifecycle import Orchestrator
            deadline = Orchestrator.confirm_deadline(e.start_ts)
            st.markdown(f"**Event {e.event_id} — {e.start_ts:%A %d/%m}, {e.start_ts:%H:%M} – {e.end_ts:%H:%M}** · "
                        f"confirm before {deadline:%d/%m %H:%M} (alerts)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max deficit", f"{e.max_deficit_kw:.1f} kW")
            m2.metric("Energy short", f"{e.shortage_energy_kwh:.1f} kWh")
            m3.metric("Share of supply", f"{100 * e.deficit_share:.1f}%")
            m4.metric("Reduction target", f"{e.recommended_reduction_pct:.0f}%")
            l, c1, c2, _ = st.columns([2, 1, 1, 2])
            level = l.selectbox("Surcharge level", list(pricing.LEVELS), index=list(pricing.LEVELS).index(e.level),
                                format_func=lambda v: f"Level {v}: +{round(pricing.LEVELS[v] * 100)}%",
                                key=f"lvl_{e.event_id}")
            if c1.button("Confirm", type="primary", key=f"confirm_{e.event_id}", width="stretch"):
                if level != e.level:
                    svc.set_level(e.event_id, level)
                with st.spinner("Building the priority list and personalised messages…"):
                    svc.confirm(e.event_id)
                st.rerun()
            if c2.button("Veto", key=f"veto_{e.event_id}", width="stretch"):
                svc.veto(e.event_id)
                st.rerun()
            st.caption(f"Forecast {e.forecast_model_version} · policy {e.decision_policy_version} · level proposed from "
                       "the share of supply missing (level 1 < 5% ≤ level 2 < 15% ≤ level 3).")


def _upcoming(up: pd.DataFrame) -> None:
    if up.empty:
        return
    st.subheader("Confirmed events")
    for e in up.itertuples():
        a, b = st.columns([4, 1])
        a.markdown(f"Event {e.event_id} — {e.start_ts:%d/%m %H:%M} – {e.end_ts:%H:%M} · level {e.level} · **{e.status}**"
                   + (" (alerts at 18:00 the day before)" if e.status == "confirmed" else ""))
        if b.button("Cancel event", key=f"cancel_{e.event_id}", width="stretch"):
            svc.cancel(e.event_id, "operator")
            st.rerun()


def _event_explorer(ev: pd.DataFrame) -> None:
    st.subheader("Events")
    if ev.empty:
        st.caption("No event yet.")
        return
    table = ev[["event_id", "start_ts", "end_ts", "level", "status", "deficit_share", "profiling_version", "device_model"]].copy()
    table["deficit_share"] = (100 * table.deficit_share).round(1)
    st.dataframe(table.rename(columns={"deficit_share": "deficit % of supply"}), hide_index=True, width="stretch", height=220)
    reached = ev[~ev.status.isin(["proposed", "vetoed", "expired"])]
    if reached.empty:
        return
    eid = st.selectbox("Event details", reached.event_id.tolist(),
                       format_func=lambda i: f"Event {i} — {reached.set_index('event_id').loc[i, 'start_ts']:%d/%m/%Y %H:%M} "
                                             f"({reached.set_index('event_id').loc[i, 'status']})")
    tg = svc.targets(eid)
    bills = svc.event_bills(event_id=eid)
    a, b, c, d = st.columns(4)
    a.metric("Households informed", len(tg))
    b.metric("Accepted", f"{(tg.response == 'accept').mean():.0%}")
    c.metric("Personalised advice", f"{tg.personalised.mean():.0%}")
    d.metric("Surcharges billed", f"{bills.surcharge_dt.sum():.2f} DT" if len(bills) else "—")
    t1, t2, t3 = st.tabs(["Priority list", "Messages sent", "Surcharges"])
    with t1:
        fig = go.Figure()
        by = tg.groupby(["segment", "response"]).size().unstack(fill_value=0)
        for resp, color in RESPONSE_COLORS.items():
            if resp in by:
                fig.add_trace(go.Bar(y=by.index, x=by[resp], name=resp.replace("_", " "), orientation="h",
                                     marker_color=color))
        fig = style(fig, "", "Households", height=300)
        fig.update_layout(barmode="stack", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
                          margin=dict(t=50))
        render_chart(fig)
        view = tg[["rank", "client_id", "segment", "acceptance_rate", "usual_window_kwh", "expected_reduction_kwh",
                   "recommended_device", "device_confidence", "band_price", "peak_price", "exempt",
                   "expected_saving_dt", "expected_surcharge_dt", "response"]].round(3)
        st.dataframe(view, hide_index=True, width="stretch", height=360)
        st.caption("Ranked by segment priority (customer profiling), then by usual consumption in the window. "
                   "Every household of the zone is informed; the rank sets the order and the level of personalisation.")
    with t2:
        nt = svc.notifications(event_id=eid)
        if nt.empty:
            st.caption("No message sent yet (alerts go out at 18:00 the day before).")
        else:
            st.dataframe(nt.groupby(["kind", "language"]).size().rename("messages").reset_index(), hide_index=True)
            st.dataframe(nt[["sent_at", "client_id", "kind", "language", "text"]], hide_index=True, width="stretch",
                         height=320)
    with t3:
        if bills.empty:
            st.caption("Bills are computed at 09:00 the day after the event.")
        else:
            st.dataframe(bills[["client_id", "window_kwh", "missing_slots", "band_price", "level", "surcharge_dt",
                                "provisional"]].sort_values("surcharge_dt", ascending=False), hide_index=True,
                         width="stretch", height=320)
            st.caption("Extra cost per household = E_window × P × s ÷ 1000. Provisional until the monthly close "
                       "(real band, 10% monthly cap).")
