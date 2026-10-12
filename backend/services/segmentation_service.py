"""Access to the customer profiling outputs (pages and orchestrator never train).

Permanent (interface)
  household_segments()                 segment, acceptance rate, comparison per household
  segment_load_profiles()              curves by segment x season x day type x slot
Before a DR event (orchestrator)
  event_targeting(event_id)            priority list for the event manager
  chatbot_context(client_id, event_id) context of one household for the chatbot
  prepare_event(start, end, deficit)   both of the above for a NEW forecast peak window
After a DR event (economics)
  event_costs(...), segment_savings(...), rewards(...)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.config import PROJECT_ROOT

PROFILING_ARTIFACT = PROJECT_ROOT / "models" / "customer_profiling" / "kmeans_v1"
TRAIN_COMMAND = "python scripts/train_customer_profiling.py"
OUTPUT_FILES = ("household_segments.csv", "segment_load_profiles.csv", "event_targeting.csv", "chatbot_context.csv",
                "event_costs.csv", "segment_savings.csv", "rewards.csv")
_DATES = {"event_targeting.csv": ["peak_start", "peak_end", "as_of"], "chatbot_context.csv": ["peak_start", "peak_end", "as_of"],
          "segment_savings.csv": ["start_ts"], "household_segments.csv": ["as_of"]}


class ProfilingArtifactError(RuntimeError):
    pass


def consumption_quantiles(energy: pd.Series) -> pd.Series:
    labels = ["Lower-use", "Typical-use", "Higher-use", "Highest-use"]
    if energy.nunique() < 4:
        return pd.Series("Typical-use", index=energy.index)
    return pd.qcut(energy.rank(method="first"), 4, labels=labels)


def profiling_available(path: Path = PROFILING_ARTIFACT) -> bool:
    return all((Path(path) / f).is_file() for f in (*OUTPUT_FILES, "metadata.json"))


@lru_cache(maxsize=16)
def _read(name: str, path: Path = PROFILING_ARTIFACT) -> pd.DataFrame:
    file = Path(path) / name
    if not file.is_file():
        raise ProfilingArtifactError(f"Customer profiling output missing ({name}). Run: {TRAIN_COMMAND}")
    return pd.read_csv(file, parse_dates=_DATES.get(name))


@lru_cache(maxsize=2)
def load_metadata(path: Path = PROFILING_ARTIFACT) -> dict:
    file = Path(path) / "metadata.json"
    if not file.is_file():
        raise ProfilingArtifactError(f"Customer profiling model not installed. Run: {TRAIN_COMMAND}")
    return json.loads(file.read_text(encoding="utf-8"))


def _filter(frame: pd.DataFrame, **keys) -> pd.DataFrame:
    for column, value in keys.items():
        if value is not None:
            frame = frame.loc[frame[column].eq(value)]
    return frame.copy()


# ---- permanent outputs (interface)
def household_segments() -> pd.DataFrame:
    return _read("household_segments.csv").copy()


def segment_load_profiles() -> pd.DataFrame:
    return _read("segment_load_profiles.csv").copy()


# ---- before a DR event (orchestrator)
def event_targeting(event_id: int | None = None) -> pd.DataFrame:
    return _filter(_read("event_targeting.csv"), event_id=event_id).sort_values(["event_id", "rank"])


def chatbot_context(client_id: str, event_id: int | None = None) -> dict:
    ctx = _filter(_read("chatbot_context.csv"), client_id=client_id, event_id=event_id)
    if ctx.empty:
        raise KeyError(f"No context for {client_id!r}")
    return ctx.sort_values("as_of").iloc[-1].to_dict()


def prepare_event(peak_start, peak_end, deficit_kwh: float, as_of=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Targeting list and chatbot context for a new peak window sent by the D+1 forecast.

    Reads the last weeks of smart-meter data available at ``as_of`` (default: the
    day before the peak at 14:00, the forecast time).
    """
    from backend.data import load_events, load_grid, load_household, load_participation
    from backend.intelligence import customer_profiling as cp

    start = pd.Timestamp(peak_start)
    as_of = pd.Timestamp(as_of) if as_of is not None else start.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    begin = min(as_of - pd.Timedelta(days=cp.USUAL_LOOKBACK_DAYS + 1), as_of.normalize().replace(day=1) - pd.Timedelta(days=7))
    segments = household_segments()
    kw30 = pd.DataFrame({cid: cp.to_half_hour_kw(cp.clean_household_power(
        load_household(cid, begin, as_of, columns=["aggregate_power_w"]))[0]) for cid in segments.client_id})
    g = load_grid().set_index("timestamp")
    g = g.loc[(g.index >= begin) & (g.index < as_of)]
    grid30 = pd.DataFrame({"temperature_c": g.temperature_c.resample(cp.SLOT).mean(),
                           "is_dr_event": g.is_dr_event.resample(cp.SLOT).max()}).reindex(kw30.index)
    acc = cp.acceptance_rates(load_participation(), load_events(), as_of)
    return (cp.event_targeting(segments, kw30, grid30, acc, peak_start, peak_end, deficit_kwh, as_of),
            cp.chatbot_context(segments, kw30, grid30, acc, peak_start, peak_end, as_of))


# ---- after a DR event (economics)
def event_costs(client_id: str | None = None, event_id: int | None = None) -> pd.DataFrame:
    return _filter(_read("event_costs.csv"), client_id=client_id, event_id=event_id)


def segment_savings(event_id: int | None = None) -> pd.DataFrame:
    return _filter(_read("segment_savings.csv"), event_id=event_id)


def rewards(client_id: str | None = None, event_id: int | None = None) -> pd.DataFrame:
    return _filter(_read("rewards.csv"), client_id=client_id, event_id=event_id)
