from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from .feature_engineering import FeatureDataset


@dataclass(frozen=True)
class ForecastResult:
    predictions: pd.DataFrame
    generated_at: pd.Timestamp
    model_name: str
    model_version: str


class Forecaster(ABC):
    @abstractmethod
    def predict(self, features: FeatureDataset) -> ForecastResult: ...

