"""Orchestrator data access, built on the project's canonical loaders (backend.data)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from backend.config import get_v3_paths
from backend.data import load_grid, load_household_info, operational_grid
from backend.orchestration import settings


@lru_cache(maxsize=1)
def grid30() -> pd.DataFrame:
    """Operational 30-min grid indexed by timestamp (households_consumption_kw, available_supply_kw, …)."""
    return operational_grid(load_grid()).set_index("timestamp")


@lru_cache(maxsize=1)
def meter30() -> pd.DataFrame:
    """Smart-meter energy per 30-min slot (kWh), one column per household. Same semantics as
    backend.data.operational_household (mean power × 0.5 h). Cached in outputs/orchestrator/."""
    if settings.METER_CACHE.is_file():
        return pd.read_parquet(settings.METER_CACHE)
    frames = []
    for path in get_v3_paths().households:
        hh = pd.read_parquet(path, columns=["timestamp", "client_id", "aggregate_power_w"])
        hh["slot"] = pd.to_datetime(hh["timestamp"]).dt.floor(settings.SLOT)
        kw = pd.to_numeric(hh["aggregate_power_w"], errors="coerce") / 1000
        mean_kw = kw.groupby([hh["slot"], hh["client_id"]]).mean()
        frames.append((mean_kw * settings.SLOT_HOURS).unstack("client_id"))
    m = pd.concat(frames, axis=1).sort_index()
    m.columns = m.columns.astype(str)
    m.index.name = "timestamp"
    settings.OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.to_parquet(settings.METER_CACHE)
    return m


def assumed_kva(row) -> int:
    """households_info.csv has no contracted power: assumption until real contracts are available."""
    if row.get("has_second_ac"):
        return 6
    if row.get("has_ac") or row.get("has_electric_water_heater"):
        return 3
    return 2


def _profiles() -> pd.DataFrame | None:
    """Segments + acceptance rates: the team's profiling model, else data/customer_profiles.csv."""
    try:
        from backend.services import segmentation_service as ss
        if ss.profiling_available():
            return ss.household_segments()[["client_id", "segment", "acceptance_rate"]]
    except ImportError:
        pass
    if settings.PROFILES_FILE.is_file():
        p = pd.read_csv(settings.PROFILES_FILE)
        p = p.rename(columns={p.columns[0]: "client_id"}) if "client_id" not in p else p
        return p[[c for c in ["client_id", "segment", "acceptance_rate"] if c in p]]
    return None


@lru_cache(maxsize=1)
def households() -> pd.DataFrame:
    """Household attributes + contracted kVA + segment and acceptance rate from customer profiling."""
    info = load_household_info().copy()
    if "contracted_kva" not in info:
        info["contracted_kva"] = info.apply(assumed_kva, axis=1)
    profiles = _profiles()
    if profiles is not None:
        info = info.merge(profiles, on="client_id", how="left")
    for col, default in [("segment", "Unknown"), ("acceptance_rate", 0.5)]:
        info[col] = info[col].fillna(default) if col in info else default
    return info.set_index("client_id")


def clear_caches() -> None:
    for f in (grid30, meter30, households):
        f.cache_clear()
    np.random.seed(None)
