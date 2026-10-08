"""Versioned on-disk registry for forecasting artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from backend.config import PROJECT_ROOT

DEFAULT_FORECAST_ARTIFACT = PROJECT_ROOT / "models" / "household_forecasting" / "energy_ttm_v1"
DEFAULT_COMMUNITY_ARTIFACT = PROJECT_ROOT / "models" / "community_dr" / "profile_ridge_v1"


class ModelArtifactError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, forecast_path: Path = DEFAULT_FORECAST_ARTIFACT,
                 community_path: Path = DEFAULT_COMMUNITY_ARTIFACT) -> None:
        self.forecast_path = Path(forecast_path)
        self.community_path = Path(community_path)

    def metadata(self, path: Path | None = None) -> dict:
        artifact = Path(path or self.forecast_path)
        metadata_path = artifact / "metadata.json"
        if not metadata_path.is_file():
            raise ModelArtifactError(
                f"Forecast model is not installed at {artifact}. Run: "
                "python scripts/train_forecasting.py"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Invalid forecast metadata: {exc}") from exc
        required = {"model_name", "model_version", "context_length", "prediction_length",
                    "frequency", "target", "model_directory", "test_start"}
        missing = required - metadata.keys()
        if missing:
            raise ModelArtifactError(f"Incompatible forecast metadata; missing {sorted(missing)}")
        if (metadata["context_length"], metadata["prediction_length"], metadata["frequency"]) != (336, 48, "30min"):
            raise ModelArtifactError("Forecast artifact is not compatible with the 336-to-48 half-hour contract.")
        model_dir = artifact / metadata["model_directory"]
        if not (model_dir / "config.json").is_file():
            raise ModelArtifactError(f"Forecast model files are missing from {model_dir}.")
        return metadata

    def available(self) -> bool:
        try:
            self.metadata()
            return True
        except ModelArtifactError:
            return False

    def community_metadata(self) -> dict:
        metadata_path = self.community_path / "metadata.json"
        model_path = self.community_path / "models.joblib"
        if not metadata_path.is_file() or not model_path.is_file():
            raise ModelArtifactError(
                f"Community replay model is not installed at {self.community_path}. Run: "
                "py -3.13 scripts/train_community_forecast.py"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("forecast_slots") != 48 or metadata.get("issuance_hour") != 14:
            raise ModelArtifactError("Community replay artifact is incompatible.")
        return metadata

    def community_available(self) -> bool:
        try:
            self.community_metadata(); return True
        except (ModelArtifactError, OSError, json.JSONDecodeError):
            return False

    def status(self) -> dict[str, str]:
        result = {"energy_ttm": str(self.forecast_path)} if self.available() else {}
        if self.community_available(): result["community_dr"] = str(self.community_path)
        return result


MODEL_REGISTRY = ModelRegistry()
