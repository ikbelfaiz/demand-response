"""Fills the notification templates with values from the event, the device recommendation and the pricing engine."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, StrictUndefined

TEMPLATES = Path(__file__).parent / "templates" / "messages.yaml"
_env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)


@lru_cache(maxsize=1)
def _load():
    return yaml.safe_load(TEMPLATES.read_text(encoding="utf-8"))


def fmt_time(ts: pd.Timestamp, lang: str) -> str:
    ts = pd.Timestamp(ts)
    if lang == "en":
        h = ts.hour % 12 or 12
        suffix = "AM" if ts.hour < 12 else "PM"
        return f"{h} {suffix}" if ts.minute == 0 else f"{h}:{ts.minute:02d} {suffix}"
    if lang == "fr":
        return f"{ts.hour}h" if ts.minute == 0 else f"{ts.hour}h{ts.minute:02d}"
    return f"{ts.hour:02d}:{ts.minute:02d}"


def fmt_num(x: float, lang: str, decimals: int = 2) -> str:
    s = f"{x:.{decimals}f}"
    return s.replace(".", ",") if lang == "fr" else s


def action_text(action: str | None, lang: str, end: str) -> str:
    t = _load()["actions"][action or "general"][lang]
    return _env.from_string(t).render(end=end)


def render(kind: str, lang: str, **values) -> str:
    return _env.from_string(_load()["messages"][kind][lang]).render(**values)


def render_for_target(kind: str, lang: str, event, target=None, *, level_pct: int,
                      projected: float | None = None, bill=None) -> tuple[str, dict]:
    """Builds the variables from DB rows and renders. Returns (text, payload stored with the notification)."""
    start, end = fmt_time(event.start_ts, lang), fmt_time(event.end_ts, lang)
    v = {"start": start, "end": end, "pct": level_pct}
    if target is not None:
        v.update(peak_price=target.peak_price, band_price=target.band_price,
                 action_text=action_text(target.action, lang, end),
                 saving=fmt_num(target.expected_saving_dt, lang),
                 expected_surcharge=fmt_num(target.expected_surcharge_dt, lang))
    if projected is not None:
        v["projected"] = fmt_num(projected, lang)
    if bill is not None:
        v.update(kwh=fmt_num(bill.window_kwh, lang), surcharge=fmt_num(bill.surcharge_dt, lang),
                 missing_slots=bill.missing_slots)
    payload = {k: (str(x) if not isinstance(x, (int, float, str)) else x) for k, x in v.items()}
    return render(kind, lang, **v), payload
