"""Energy-TTM model loading and leakage-safe half-hour inference."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_features import CONTEXT_SLOTS, HORIZON_SLOTS, normalize_context
from .model_registry import DEFAULT_FORECAST_ARTIFACT, ModelArtifactError, ModelRegistry

FORECAST_COLUMNS = ["forecast_timestamp", "predicted_power_kw", "predicted_energy_kwh"]


class ForecastRuntimeError(RuntimeError):
    pass


def build_history_at_issuance(operational: pd.DataFrame, issuance_time: object) -> pd.DataFrame:
    """Only observations strictly available before forecast issuance."""
    return operational.loc[operational.timestamp < pd.Timestamp(issuance_time)].copy()


def empty_forecast() -> pd.DataFrame:
    return pd.DataFrame(columns=FORECAST_COLUMNS)


class EnergyTTMForecaster:
    """Thin adapter around the notebook's TinyTimeMixerForPrediction model."""

    def __init__(self, artifact_path: Path | str = DEFAULT_FORECAST_ARTIFACT) -> None:
        self.artifact_path = Path(artifact_path)
        registry = ModelRegistry(self.artifact_path)
        self.metadata = registry.metadata()
        try:
            import torch
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        except ImportError as exc:
            raise ForecastRuntimeError(
                "Energy-TTM runtime is unavailable. Use Python 3.10-3.13 and install requirements.txt."
            ) from exc
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.artifact_path / self.metadata["model_directory"]
        try:
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                str(model_path), local_files_only=True
            ).to(self.device).eval()
        except Exception as exc:
            raise ModelArtifactError(f"Could not load Energy-TTM artifact at {model_path}: {exc}") from exc
        if int(self.model.config.context_length) != CONTEXT_SLOTS or int(self.model.config.prediction_length) != HORIZON_SLOTS:
            raise ModelArtifactError("Loaded Energy-TTM dimensions do not match 336 inputs and 48 outputs.")

    def predict(self, context_kw: np.ndarray) -> np.ndarray:
        scaled, mean, std = normalize_context(context_kw)
        tensor = self._torch.tensor(scaled, dtype=self._torch.float32, device=self.device).reshape(1, CONTEXT_SLOTS, 1)
        with self._torch.inference_mode():
            output = self.model(tensor).prediction_outputs[0, :HORIZON_SLOTS, 0]
        prediction = output.detach().cpu().numpy() * (std + 1e-8) + mean
        prediction = np.clip(prediction.astype(np.float32), 0, None)
        if prediction.shape != (HORIZON_SLOTS,) or not np.isfinite(prediction).all():
            raise ForecastRuntimeError("Energy-TTM returned an invalid prediction.")
        return prediction


@lru_cache(maxsize=2)
def load_forecaster(artifact_path: str = str(DEFAULT_FORECAST_ARTIFACT)) -> EnergyTTMForecaster:
    return EnergyTTMForecaster(artifact_path)


def predict_day_ahead(context_kw: np.ndarray, artifact_path: Path | str = DEFAULT_FORECAST_ARTIFACT) -> np.ndarray:
    return load_forecaster(str(artifact_path)).predict(context_kw)
