from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from .event_detection import DetectionResult


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    evaluated_at: pd.Timestamp
    evaluation_version: str


class DetectionEvaluator(ABC):
    """Evaluate frozen detections against explicitly supplied held-out labels."""
    @abstractmethod
    def evaluate(self, detections: DetectionResult, held_out_labels: pd.DataFrame) -> EvaluationResult: ...
