"""DR chat assistant: keyword intents → answers built from the orchestrator database.

The assistant never invents numbers: prices, savings and surcharges are the ones the pricing engine
stored on the household's EventTarget / EventBill. Responses (accept / decline / later) are written
with Orchestrator.record_response, exactly like the buttons of the notification card.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, StrictUndefined

from backend.intelligence import pricing
from backend.orchestration.notifications.render import action_text, fmt_num, fmt_time

CHAT = Path(__file__).parent / "templates" / "chat.yaml"
_env = Environment(undefined=StrictUndefined, autoescape=False)
# order matters: a response word wins over a question word ("ok, what should I do?" → accept)
INTENT_ORDER = ["accept", "decline", "later", "why", "compare", "history", "when", "price", "cost", "action", "help"]
DAYS = {"en": "%A %d %B", "fr": "%d/%m", "ar": "%d/%m"}


@lru_cache(maxsize=1)
def _chat() -> dict:
    return yaml.safe_load(CHAT.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return f" {re.sub(r'[!?.,;:]+', ' ', text)} "


def detect_intent(text: str) -> str:
    t = _norm(text)
    for intent in INTENT_ORDER:
        for kw in _chat()["keywords"][intent]:
            k = _norm(str(kw)).strip()
            # short words must match whole words ("ok" ≠ "book", "لا" ≠ "لاحقا"); longer keywords match anywhere
            if (f" {k} " in t) if len(k) <= 3 else (k in t):
                return intent
    return "help"


def answer(key: str, lang: str, **values) -> str:
    return _env.from_string(_chat()["answers"][key][lang]).render(**values)


@dataclass
class ChatContext:
    """What the assistant knows about one household at the demo time `now`."""
    now: pd.Timestamp
    lang: str
    event: object | None        # DREvent the household is targeted by (upcoming or running), or None
    target: object | None       # its EventTarget
    last_bill: object | None    # most recent EventBill (with .event_start)
    n_events: int
    n_accepted: int
    total_surcharge: float
    comparison_message: str | None


def reply(text: str, ctx: ChatContext, record_response=None) -> tuple[str, str]:
    """Returns (intent, answer text). `record_response(response)` stores accept / decline / later."""
    intent = detect_intent(text)
    lang, ev, t = ctx.lang, ctx.event, ctx.target
    if intent == "help":
        return intent, answer("help", lang)
    if intent == "history":
        return intent, answer("history", lang, n=ctx.n_events, accepted=ctx.n_accepted,
                              total=fmt_num(ctx.total_surcharge, lang))
    if intent == "compare":
        msg = ctx.comparison_message or (t.comparison_message if t is not None else None)
        return intent, answer("compare", lang, message=msg) if msg else answer("compare_none", lang)
    if intent == "cost" and (ev is None or t is None) and ctx.last_bill is not None:
        b = ctx.last_bill
        return intent, answer("cost_billed", lang, day=pd.Timestamp(b.event_start).strftime(DAYS[lang]),
                              kwh=fmt_num(b.window_kwh, lang), surcharge=fmt_num(b.surcharge_dt, lang))
    if ev is None or t is None:
        return intent, answer("no_event", lang)

    start, end = fmt_time(ev.start_ts, lang), fmt_time(ev.end_ts, lang)
    pct = round(pricing.LEVELS[ev.level] * 100)
    if intent in ("accept", "decline", "later"):
        if ctx.now >= pd.Timestamp(ev.start_ts):
            return intent, answer("too_late", lang)
        if record_response is not None:
            record_response(intent)
        return intent, answer(intent, lang, start=start, end=end)
    if intent == "when":
        return intent, answer("when", lang, day=pd.Timestamp(ev.start_ts).strftime(DAYS[lang]), start=start, end=end)
    if intent == "price":
        if t.exempt:
            return intent, answer("price_exempt", lang, band_price=t.band_price)
        return intent, answer("price", lang, peak_price=t.peak_price, band_price=t.band_price, pct=pct)
    if intent == "action":
        act = action_text(t.action, lang, end)
        if t.personalised and t.expected_saving_dt > 0:
            return intent, answer("action", lang, action_text=act, saving=fmt_num(t.expected_saving_dt, lang))
        return intent, answer("action_general", lang, action_text=act, band_price=t.band_price, peak_price=t.peak_price)
    if intent == "cost":
        return intent, answer("cost", lang, usual_kwh=fmt_num(t.usual_window_kwh, lang),
                              expected_surcharge=fmt_num(t.expected_surcharge_dt, lang))
    if intent == "why":
        return intent, answer("why", lang, share=fmt_num(100 * (ev.deficit_share or 0), lang, 1))
    return intent, answer("help", lang)
