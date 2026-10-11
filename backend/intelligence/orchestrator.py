"""Separation point for forecasting, proposal, participation, and evaluation.

The DR orchestrator itself lives in backend/orchestration/ (event manager, pricing, notifications).
It consumes the community forecast + decision policy and turns recommendations into household
notifications and peak-price bills. Households act by hand: there is no operational activation.
"""
from dataclasses import dataclass
from .model_registry import MODEL_REGISTRY


@dataclass(frozen=True)
class OrchestratorStatus:
    forecast_model: bool = False
    flexibility_model: bool = False
    decision_model: bool = False
    operational_activation: bool = False
    notification_engine: bool = False
    pricing_engine: bool = False


def status() -> OrchestratorStatus:
    return OrchestratorStatus(forecast_model=MODEL_REGISTRY.available(),
                              decision_model=MODEL_REGISTRY.community_available(),
                              notification_engine=True, pricing_engine=True)


def build_orchestrator(db_url: str | None = None, **kwargs):
    """Create (if needed) the orchestrator database and return an Orchestrator wired to the team's models."""
    from sqlalchemy import inspect

    from backend.orchestration import settings
    from backend.orchestration.lifecycle import Orchestrator
    from backend.orchestration.store import get_engine, get_sessionmaker, init_db
    url = db_url or settings.DATABASE_URL
    settings.OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine(url)
    if not inspect(engine).has_table("clients"):
        engine = init_db(url)
    return Orchestrator(get_sessionmaker(engine), **kwargs)
