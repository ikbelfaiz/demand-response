from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from .event_detection import DetectionResult


@dataclass(frozen=True)
class ActivationDecision:
    decisions: pd.DataFrame
    generated_at: pd.Timestamp
    policy_name: str
    policy_version: str


class DecisionEngine(ABC):
    """Separate operational activation policy from condition detection."""
    @abstractmethod
    def decide(self, detections: DetectionResult) -> ActivationDecision: ...


@dataclass(frozen=True)
class FlexibilityEstimate:
    estimates: pd.DataFrame
    generated_at: pd.Timestamp
    model_name: str
    model_version: str


class FlexibilityEstimator(ABC):
    @abstractmethod
    def estimate(self, features) -> FlexibilityEstimate: ...


@dataclass(frozen=True)
class OptimizationPlan:
    actions: pd.DataFrame
    generated_at: pd.Timestamp
    optimizer_name: str
    optimizer_version: str


class DROptimizer(ABC):
    @abstractmethod
    def optimize(self, decision: ActivationDecision, flexibility: FlexibilityEstimate) -> OptimizationPlan: ...
