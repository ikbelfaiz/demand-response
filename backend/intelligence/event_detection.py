from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from .feature_engineering import FeatureDataset


@dataclass(frozen=True)
class DetectionResult:
    candidates: pd.DataFrame
    generated_at: pd.Timestamp
    model_name: str
    model_version: str
    interpretation: str = "Candidate grid conditions; not DR activation decisions."


class EventDetector(ABC):
    @abstractmethod
    def detect(self, features: FeatureDataset) -> DetectionResult: ...

