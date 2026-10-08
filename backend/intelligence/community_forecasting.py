"""Causal day-ahead profile models for community demand and supply components."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROJECT_ROOT, get_v3_paths

ISSUE_HOUR = 14
HORIZON_SLOTS = 48
FREQUENCY = "30min"
COMMUNITY_ARTIFACT = PROJECT_ROOT / "models" / "community_dr" / "profile_ridge_v1"
ORIGINAL_GRID = PROJECT_ROOT / "data" / "original_v3_backup" / "grid_1min.csv"
MODEL_COLUMNS = ("households_consumption_kw", "steg_production_kw", "pv_production_kw")


class CommunityModelError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_causal_operational_grid(path_text: str = str(ORIGINAL_GRID)) -> pd.DataFrame:
    """Aggregate original sensor values without two-sided interpolation.

    A slot is usable with at least 80% of its source minutes. Coverage columns retain
    provenance so evaluation can distinguish complete from partially observed slots.
    """
    raw = pd.read_csv(path_text, parse_dates=["timestamp"], low_memory=False).set_index("timestamp").sort_index()
    index = raw.resample(FREQUENCY).size().index
    out = pd.DataFrame({"timestamp": index})
    for column in MODEL_COLUMNS + ("temperature_c",):
        numeric = pd.to_numeric(raw[column], errors="coerce")
        count = numeric.resample(FREQUENCY).count()
        mean = numeric.resample(FREQUENCY).mean()
        coverage = count / 30.0
        out[column] = mean.where(coverage >= 0.8).to_numpy()
        out[f"{column}_coverage"] = coverage.to_numpy()
    for column in ("is_holiday", "is_ramadan", "is_dr_event", "is_dr_peak"):
        numeric = pd.to_numeric(raw[column], errors="coerce")
        out[column] = numeric.resample(FREQUENCY).max().to_numpy()
        out[f"{column}_coverage"] = (numeric.resample(FREQUENCY).count()/30.0).to_numpy()
    out["available_supply_kw"] = out.steg_production_kw + out.pv_production_kw
    out["margin_kw"] = out.available_supply_kw - out.households_consumption_kw
    return out


def issue_timestamp(target_date: object) -> pd.Timestamp:
    return pd.Timestamp(target_date).normalize() - pd.DateOffset(days=1) + pd.DateOffset(hours=ISSUE_HOUR)


def forecast_index(target_date: object) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(target_date).normalize(), periods=HORIZON_SLOTS, freq=FREQUENCY)


def _profile(series: pd.Series, day: pd.Timestamp) -> np.ndarray:
    index = pd.date_range(day.normalize(), periods=HORIZON_SLOTS, freq=FREQUENCY)
    return series.reindex(index).to_numpy(dtype=np.float64)


def build_causal_features(frame: pd.DataFrame, target_date: object, column: str) -> np.ndarray:
    """Build features using sensor values strictly before D-1 14:00."""
    if column not in MODEL_COLUMNS:
        raise KeyError(column)
    day, cutoff = pd.Timestamp(target_date).normalize(), issue_timestamp(target_date)
    series = frame.set_index("timestamp")[column].sort_index()
    recent_index = pd.date_range(cutoff - pd.offsets.Hour(24), periods=48, freq=FREQUENCY)
    if recent_index[-1] >= cutoff:
        raise AssertionError("Recent feature window crosses forecast issuance.")
    recent = series.reindex(recent_index).to_numpy(dtype=np.float64)
    lag2 = _profile(series, day - pd.DateOffset(days=2))
    lag7 = _profile(series, day - pd.DateOffset(days=7))
    prior_profiles = np.vstack([_profile(series, day - pd.DateOffset(days=offset)) for offset in range(2, 9)])
    mean_profile = prior_profiles.mean(axis=0)
    dow = np.zeros(7, dtype=float); dow[day.dayofweek] = 1.0
    month_angle = 2 * np.pi * (day.month - 1) / 12
    # Holiday and Ramadan are deterministic calendar flags known at issuance.
    target_row = frame.loc[frame.timestamp.eq(day), ["is_holiday", "is_ramadan"]]
    calendar = target_row.iloc[0].to_numpy(dtype=float) if not target_row.empty else np.zeros(2)
    features = np.concatenate([recent, lag2, lag7, mean_profile, dow,
                               [np.sin(month_angle), np.cos(month_angle)], calendar])
    if not np.isfinite(features).all():
        raise ValueError(f"Incomplete causal feature history for {column} on {day.date()}.")
    return features.astype(np.float32)


def build_target(frame: pd.DataFrame, target_date: object, column: str) -> np.ndarray:
    values = _profile(frame.set_index("timestamp")[column].sort_index(), pd.Timestamp(target_date))
    if values.shape != (48,) or not np.isfinite(values).all():
        raise ValueError(f"Incomplete target for {column} on {pd.Timestamp(target_date).date()}.")
    return values.astype(np.float32)


def build_training_matrix(frame: pd.DataFrame, dates, column: str,
                          *, excluded_dates=frozenset()) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    features, targets, accepted = [], [], []
    excluded = {pd.Timestamp(value).normalize() for value in excluded_dates}
    for date in pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize():
        if date in excluded:
            continue
        try:
            feature = build_causal_features(frame, date, column)
            target = build_target(frame, date, column)
            features.append(feature)
            targets.append(target)
            accepted.append(date)
        except ValueError:
            continue
    if not features:
        raise ValueError(f"No eligible {column} windows.")
    return np.asarray(features, np.float32), np.asarray(targets, np.float32), pd.DatetimeIndex(accepted)


@dataclass
class ProfileModel:
    column: str
    scaler: object
    regressor: object

    def predict(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, np.float32).reshape(1, -1))
        result = self.regressor.predict(transformed)[0]
        return np.clip(np.asarray(result, np.float32), 0, None)


class CommunityForecastBundle:
    def __init__(self, artifact_path: Path | str = COMMUNITY_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        metadata_path = self.artifact_path / "metadata.json"
        models_path = self.artifact_path / "models.joblib"
        if not metadata_path.is_file() or not models_path.is_file():
            raise CommunityModelError(
                f"Community replay models are unavailable at {self.artifact_path}. "
                "Run: py -3.13 scripts/train_community_forecast.py"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("forecast_slots") != 48 or self.metadata.get("issuance_hour") != 14:
            raise CommunityModelError("Community model artifact is incompatible with the replay contract.")
        try:
            import joblib
            self.models: dict[str, ProfileModel] = joblib.load(models_path)
        except Exception as exc:
            raise CommunityModelError(f"Could not load community models: {exc}") from exc
        if set(self.models) != set(MODEL_COLUMNS):
            raise CommunityModelError("Community artifact does not contain demand, STEG, and PV models.")

    def predict_day(self, frame: pd.DataFrame, target_date: object) -> pd.DataFrame:
        index = forecast_index(target_date)
        output = pd.DataFrame({"timestamp": index})
        names = {"households_consumption_kw":"predicted_demand_kw",
                 "steg_production_kw":"predicted_steg_kw", "pv_production_kw":"predicted_pv_kw"}
        for column, model in self.models.items():
            output[names[column]] = model.predict(build_causal_features(frame, target_date, column))
        output["predicted_supply_kw"] = output.predicted_steg_kw + output.predicted_pv_kw
        output["predicted_margin_kw"] = output.predicted_supply_kw - output.predicted_demand_kw
        return output


@lru_cache(maxsize=2)
def load_community_bundle(path_text: str = str(COMMUNITY_ARTIFACT)) -> CommunityForecastBundle:
    return CommunityForecastBundle(path_text)


def artifact_status(path: Path | str = COMMUNITY_ARTIFACT) -> tuple[bool, str]:
    try:
        bundle = CommunityForecastBundle(path)
        return True, f"{bundle.metadata['model_name']} {bundle.metadata['model_version']}"
    except Exception as exc:
        return False, str(exc)
