"""Efficient, read-only access to the Demand Response v3 source files."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from backend.config import V3Paths, get_v3_paths

HOUSEHOLD_COLUMNS = [
    "timestamp", "client_id", "aggregate_power_w", "ac_power_w",
    "water_heater_power_w", "washing_machine_power_w",
]


def assert_v3_files(paths: V3Paths | None = None) -> None:
    missing = [str(p) for p in (paths or get_v3_paths()).inventory().values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing v3 files: " + ", ".join(missing))


@lru_cache(maxsize=4)
def _load_grid(path_text: str, size: int, modified: int) -> pd.DataFrame:
    path = Path(path_text)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_grid() -> pd.DataFrame:
    path=get_v3_paths().grid; stat=path.stat()
    return _load_grid(str(path),stat.st_size,stat.st_mtime_ns)


@lru_cache(maxsize=1)
def load_household_info() -> pd.DataFrame:
    return pd.read_csv(get_v3_paths().household_info)


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    return pd.read_csv(get_v3_paths().events, parse_dates=["start_ts", "end_ts"])


@lru_cache(maxsize=1)
def load_participation() -> pd.DataFrame:
    return pd.read_csv(get_v3_paths().participation)


def load_ground_truth_for_evaluation() -> pd.DataFrame:
    """Research-only accessor; never imported by operational baseline code."""
    return pd.read_csv(get_v3_paths().ground_truth)


def _household_path(client_id: str) -> Path:
    number = int(client_id.removeprefix("C"))
    if not 1 <= number <= 50:
        raise KeyError(f"Unknown household {client_id!r}")
    return get_v3_paths().households[0 if number <= 25 else 1]


def load_household(
    client_id: str,
    start: object | None = None,
    end: object | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read one household with parquet predicate and column pushdown."""
    wanted = list(dict.fromkeys(columns or HOUSEHOLD_COLUMNS))
    for required in ("timestamp", "client_id"):
        if required not in wanted:
            wanted.insert(0, required)
    filters: list[tuple[str, str, object]] = [("client_id", "==", client_id)]
    if start is not None:
        filters.append(("timestamp", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("timestamp", "<", pd.Timestamp(end)))
    frame = pd.read_parquet(_household_path(client_id), columns=wanted, filters=filters, engine="pyarrow")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def filter_time(frame: pd.DataFrame, start: object, end: object, column: str = "timestamp") -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return frame.loc[(frame[column] >= start_ts) & (frame[column] < end_ts)].copy()
