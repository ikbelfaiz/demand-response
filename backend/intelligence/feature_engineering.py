from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class FeatureDataset:
    frame: pd.DataFrame
    generated_at: pd.Timestamp
    feature_version: str
    label_column: str | None = None


def build_monitoring_features(data: pd.DataFrame, feature_version: str = "v1") -> FeatureDataset:
    """Build transparent non-predictive inputs; labels remain optional and separate."""
    required = ["timestamp", "zone_demand_mw", "system_production_mw", "zone_pv_production_mw", "temperature_c"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing feature channels: {', '.join(missing)}")
    frame = data[required].copy()
    frame["hour"] = frame.timestamp.dt.hour
    frame["day_of_week"] = frame.timestamp.dt.dayofweek
    return FeatureDataset(frame=frame, generated_at=pd.Timestamp.now(tz="UTC"), feature_version=feature_version)

