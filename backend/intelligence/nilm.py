"""Trusted TCN NILM artifact loading and aggregate-only inference primitives."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import hashlib

import torch

from backend.config import DATA_DIR, PROJECT_ROOT, V3_PATHS
from nilm_research.checkpoint import load_checkpoint
from nilm_research.model import APPLIANCES, CausalMultiTaskTCN

TARGET_COLUMNS = ["ac_power_w", "water_heater_power_w", "washing_machine_power_w"]
EXPECTED_APPLIANCES = tuple(APPLIANCES)


class NILMArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class NILMRuntime:
    model: torch.nn.Module
    config: MappingProxyType
    transforms: MappingProxyType
    metadata: MappingProxyType
    device: torch.device


def artifact_identity(path: str | Path) -> tuple[str, int, int]:
    target = Path(path)
    stat = target.stat()
    return str(target.resolve()), stat.st_size, stat.st_mtime_ns


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise NILMArtifactError("CUDA was requested for NILM, but CUDA is unavailable")
    return torch.device(requested)


def _validate(payload: dict, model: torch.nn.Module) -> None:
    if payload.get("model_kind") != "tcn" or not isinstance(model, CausalMultiTaskTCN):
        raise NILMArtifactError("Configured NILM artifact is not a causal TCN checkpoint")
    transforms = payload.get("transforms", {})
    if tuple(transforms.get("appliance_order", ())) != EXPECTED_APPLIANCES:
        raise NILMArtifactError(f"Incompatible appliance order: {transforms.get('appliance_order')!r}")
    if transforms.get("units") != "watts":
        raise NILMArtifactError("NILM checkpoint output units must be watts")
    config = payload.get("config", {})
    if int(config.get("data", {}).get("context_length", 0)) != 256:
        raise NILMArtifactError("NILM checkpoint must use a 256-minute context")
    expected = {"channels": 64, "dilations": [1, 2, 4, 8, 16, 32], "dropout": 0.1}
    for key, value in expected.items():
        if config.get("model", {}).get(key) != value:
            raise NILMArtifactError(f"Incompatible TCN {key}: {config.get('model', {}).get(key)!r}")
    if model.receptive_field != 253:
        raise NILMArtifactError(f"Incompatible TCN receptive field: {model.receptive_field}")
    with torch.inference_mode():
        output = model(torch.zeros(1, 2, 256))
    if output["power_w"].shape != (1, 3) or output["activity_probability"].shape != (1, 3):
        raise NILMArtifactError("NILM checkpoint outputs must have shape [batch, 3]")


def _runtime_config(saved: dict) -> dict:
    """Keep learned policy but bind data to this checkout, never saved absolute paths."""
    cfg = deepcopy(saved)
    cfg["data"].update({
        "household_paths": [str(path) for path in V3_PATHS.households],
        "cleaned_household_paths": [
            str(DATA_DIR / "cleaned_v3" / "households_1min_C001_C025_cleaned.parquet"),
            str(DATA_DIR / "cleaned_v3" / "households_1min_C026_C050_cleaned.parquet"),
        ],
        "household_info_path": str(V3_PATHS.household_info),
        "grid_path": str(V3_PATHS.grid),
        "cleaning_manifest_path": str(DATA_DIR / "cleaned_v3" / "cleaning_manifest.json"),
    })
    return cfg


@lru_cache(maxsize=4)
def _load_runtime(identity: tuple[str, int, int], requested_device: str) -> NILMRuntime:
    path = Path(identity[0])
    try:
        device = _device(requested_device)
        model, payload = load_checkpoint(path, "cpu")
        _validate(payload, model)
        model.to(device).eval()
        cfg = _runtime_config(payload["config"])
        transforms = deepcopy(payload["transforms"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        metadata = {
            "model_identifier": f"causal-multitask-tcn-{digest}",
            "model_kind": "TCN", "checkpoint_sha256_prefix": digest,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "best_validation_loss": float(payload.get("best_validation_loss", float("nan"))),
            "context_length": int(cfg["data"]["context_length"]),
            "sampling_interval_minutes": 1,
            "appliance_order": list(EXPECTED_APPLIANCES), "units": "watts",
            "activity_thresholds_w": list(transforms["activity_thresholds_w"]),
            "execution_device": str(device),
        }
        return NILMRuntime(model, MappingProxyType(cfg), MappingProxyType(transforms),
                           MappingProxyType(metadata), device)
    except NILMArtifactError:
        raise
    except Exception as exc:
        raise NILMArtifactError(f"Unable to load trusted NILM checkpoint {path}: {exc}") from exc


def load_nilm_runtime(path: str | Path, requested_device: str = "auto") -> NILMRuntime:
    target = Path(path).resolve()
    if not target.is_relative_to(PROJECT_ROOT.resolve()):
        raise NILMArtifactError("NILM checkpoint must be a trusted file inside the repository")
    if not target.is_file():
        raise NILMArtifactError(f"NILM checkpoint not found: {target}")
    return _load_runtime(artifact_identity(target), requested_device)


def estimate_appliances(*args, **kwargs):
    """Compatibility entry point; delegates to the application NILM service."""
    from backend.services.nilm_service import predict_appliances
    return predict_appliances(*args, **kwargs)
