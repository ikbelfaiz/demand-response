from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import PartitionData, season
from .model import APPLIANCES


@dataclass
class MetricBucket:
    count: int = 0
    absolute_error_w: float = 0.0
    predicted_energy_kwh: float = 0.0
    true_energy_kwh: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    hourly_count: int = 0
    hourly_absolute_error_kwh: float = 0.0
    hourly_coverage_sum: float = 0.0

    def update_points(self, predicted: np.ndarray, true: np.ndarray, predicted_active: np.ndarray,
                      true_active: np.ndarray) -> None:
        self.count += len(true)
        self.absolute_error_w += float(np.abs(predicted - true).sum())
        self.predicted_energy_kwh += float(predicted.sum() / 60000.0)
        self.true_energy_kwh += float(true.sum() / 60000.0)
        self.tp += int((predicted_active & true_active).sum())
        self.fp += int((predicted_active & ~true_active).sum())
        self.fn += int((~predicted_active & true_active).sum())

    def update_hours(self, predicted: np.ndarray, true: np.ndarray, timestamps: np.ndarray) -> None:
        if not len(true):
            return
        hours = timestamps.astype("datetime64[h]")
        for hour in np.unique(hours):
            mask = hours == hour
            coverage = int(mask.sum())
            pred_kwh = float(predicted[mask].sum() / 60000.0)
            true_kwh = float(true[mask].sum() / 60000.0)
            self.hourly_count += 1
            self.hourly_absolute_error_kwh += abs(pred_kwh - true_kwh)
            self.hourly_coverage_sum += coverage / 60.0

    def result(self) -> dict:
        precision_den = self.tp + self.fp
        recall_den = self.tp + self.fn
        precision = self.tp / precision_den if precision_den else None
        recall = self.tp / recall_den if recall_den else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        rel = ((self.predicted_energy_kwh - self.true_energy_kwh) / self.true_energy_kwh
               if self.true_energy_kwh > 0 else None)
        return {
            "valid_observations": self.count,
            "power_mae_w": self.absolute_error_w / self.count if self.count else None,
            "predicted_total_energy_kwh": self.predicted_energy_kwh,
            "true_total_energy_kwh": self.true_energy_kwh,
            "relative_total_energy_error": rel,
            "activity_precision": precision, "activity_recall": recall, "activity_f1": f1,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "hourly_energy_mae_kwh": self.hourly_absolute_error_kwh / self.hourly_count if self.hourly_count else None,
            "hourly_intervals": self.hourly_count,
            "mean_hourly_coverage": self.hourly_coverage_sum / self.hourly_count if self.hourly_count else None,
        }


class MetricsAccumulator:
    def __init__(self, probability_threshold: float = 0.5):
        self.threshold = probability_threshold
        self.buckets: dict[tuple[str, str, str, str], MetricBucket] = {}

    def _update(self, appliance: str, household: str, dimension: str, value: str,
                predicted: np.ndarray, true: np.ndarray, probability: np.ndarray,
                true_active: np.ndarray, timestamps: np.ndarray) -> None:
        key = appliance, household, dimension, value
        bucket = self.buckets.setdefault(key, MetricBucket())
        bucket.update_points(predicted, true, probability >= self.threshold, true_active)
        bucket.update_hours(predicted, true, timestamps)

    def add_partition(self, part: PartitionData, predicted_w: np.ndarray, probabilities: np.ndarray,
                      valid_endpoint: np.ndarray, population: str) -> None:
        months = part.timestamps.astype("datetime64[M]").astype(np.int64) % 12 + 1
        seasons = np.asarray([season(int(x)) for x in months], dtype=object)
        dimensions: list[tuple[str, np.ndarray]] = [
            ("overall", np.full(len(months), "all", dtype=object)),
            ("dr", np.where(part.dr_event, "dr_event", "ordinary")),
            ("season", seasons),
            ("population", np.full(len(months), population, dtype=object)),
        ]
        for j, appliance in enumerate(APPLIANCES):
            valid = valid_endpoint & part.target_valid[:, j] & np.isfinite(predicted_w[:, j])
            state = np.where(part.activity[:, j] > 0.5, "active", "inactive")
            for dimension, labels in dimensions + [("state", state)]:
                for value in np.unique(labels[valid]):
                    mask = valid & (labels == value)
                    for household in (part.client_id, "ALL"):
                        self._update(appliance, household, dimension, str(value), predicted_w[mask, j],
                                     part.targets_w[mask, j], probabilities[mask, j],
                                     part.activity[mask, j].astype(bool), part.timestamps[mask])

    def rows(self, split: str, model_kind: str) -> list[dict]:
        rows = []
        for (appliance, household, dimension, value), bucket in sorted(self.buckets.items()):
            rows.append({"model": model_kind, "split": split, "appliance": appliance,
                         "household": household, "stratum": dimension, "stratum_value": value,
                         **bucket.result()})
        return rows

