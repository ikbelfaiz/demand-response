"""Typed contracts shared by data, service, and future intelligence layers."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationReport:
    source: str
    row_count: int
    start: object | None
    end: object | None
    duplicate_keys: int
    missing_values: dict[str, int] = field(default_factory=dict)
    invalid_records: int = 0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastSlot:
    timestamp: object
    predicted_demand_kw: float
    p50_kw: float
    p90_kw: float
    forecast_supply_kw: float | None
    expected_margin_kw: float | None
    peak_probability: float | None


@dataclass(frozen=True)
class ApplianceEstimate:
    timestamp: object
    appliance: str
    estimated_power_kw: float
    confidence: float | None
    provenance: str = "model_estimate"
