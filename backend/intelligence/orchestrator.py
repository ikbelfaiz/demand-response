"""Separation point for forecasting, proposal, participation, and evaluation."""
from dataclasses import dataclass
from .model_registry import MODEL_REGISTRY


@dataclass(frozen=True)
class OrchestratorStatus:
    forecast_model: bool = False
    flexibility_model: bool = False
    decision_model: bool = False
    operational_activation: bool = False


def status() -> OrchestratorStatus:
    return OrchestratorStatus(forecast_model=MODEL_REGISTRY.available())
