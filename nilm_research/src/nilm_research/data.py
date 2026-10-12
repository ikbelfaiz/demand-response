from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import APPLIANCES

TARGET_COLUMNS = ("ac_power_w", "water_heater_power_w", "washing_machine_power_w")
FLAG_COLUMNS = tuple(f"{x}_was_imputed" for x in ("aggregate_power_w", *TARGET_COLUMNS))


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required; install nilm_research/requirements.txt") from exc
    return duckdb


def _sql_list(paths: Sequence[str]) -> str:
    return "[" + ",".join("'" + str(x).replace("'", "''") + "'" for x in paths) + "]"


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


@dataclass
class PartitionData:
    client_id: str
    split: str
    timestamps: np.ndarray
    aggregate_w: np.ndarray
    targets_w: np.ndarray
    aggregate_imputed: np.ndarray
    target_imputed: np.ndarray
    dr_event: np.ndarray
    bad_context_point: np.ndarray
    target_valid: np.ndarray
    inputs: np.ndarray
    activity: np.ndarray
    frozen: np.ndarray
    aggregate_spike: np.ndarray
    target_contaminated: np.ndarray
    timestamp_gap: np.ndarray
    aggregate_nonfinite: np.ndarray


class NILMRepository:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        data = cfg["data"]
        self.household_paths = data["household_paths"]
        self.cleaned_paths = data["cleaned_household_paths"]
        self.info_path = data["household_info_path"]
        self.grid_path = data["grid_path"]

    def connect(self):
        con = _duckdb().connect(":memory:")
        con.execute("PRAGMA threads=4")
        return con

    def panel_households(self) -> list[str]:
        h = _sql_list(self.household_paths)
        with self.connect() as con:
            from_data = [x[0] for x in con.execute(
                f"SELECT client_id FROM read_parquet({h}) GROUP BY client_id "
                "HAVING count(ac_power_w)>0 AND count(water_heater_power_w)>0 "
                "AND count(washing_machine_power_w)>0 ORDER BY client_id"
            ).fetchall()]
            meta = [x[0] for x in con.execute(
                f"SELECT client_id FROM read_csv_auto('{self.info_path}') WHERE has_submeter ORDER BY client_id"
            ).fetchall()]
        if from_data != meta:
            raise ValueError(f"Panel mismatch: data={from_data}, metadata={meta}")
        return from_data

    def verify(self) -> dict:
        h = _sql_list(self.household_paths)
        c = _sql_list(self.cleaned_paths)
        with self.connect() as con:
            rows = con.execute(f"SELECT count(*),count(distinct client_id),min(timestamp),max(timestamp) FROM read_parquet({h})").fetchone()
            cols = [x[0] for x in con.execute(f"DESCRIBE SELECT * FROM read_parquet({h})").fetchall()]
            timeline = con.execute(
                f"WITH x AS (SELECT timestamp,client_id,lag(timestamp) OVER(PARTITION BY client_id ORDER BY timestamp) p FROM read_parquet({h})) "
                "SELECT count(*) FILTER(WHERE p IS NOT NULL AND timestamp-p<>INTERVAL 1 MINUTE),"
                "count(*)-count(distinct (client_id,timestamp)) FROM x"
            ).fetchone()
            flags = con.execute(
                f"SELECT sum(aggregate_power_w_was_imputed),sum(ac_power_w_was_imputed),"
                f"sum(water_heater_power_w_was_imputed),sum(washing_machine_power_w_was_imputed) FROM read_parquet({c})"
            ).fetchone()
        expected = ["timestamp", "client_id", "aggregate_power_w", *TARGET_COLUMNS]
        return {
            "rows": rows[0], "households": rows[1], "start": str(rows[2]), "end": str(rows[3]),
            "columns": cols, "expected_columns_present": all(x in cols for x in expected),
            "bad_intervals": timeline[0], "duplicate_keys": timeline[1],
            "imputed_counts": dict(zip(FLAG_COLUMNS, map(int, flags))),
            "panel_households": self.panel_households(),
        }

    def load_raw_partition(self, client_id: str, split: str, bounds: dict) -> dict[str, np.ndarray]:
        start, end = bounds[split]
        return self.load_raw_range(client_id, start, end)

    def load_raw_range(self, client_id: str, start: str, end: str) -> dict[str, np.ndarray]:
        h, c = _sql_list(self.household_paths), _sql_list(self.cleaned_paths)
        grid = str(self.grid_path).replace("'", "''")
        query = f"""
            SELECT h.timestamp,h.aggregate_power_w,h.ac_power_w,h.water_heater_power_w,h.washing_machine_power_w,
                   f.aggregate_power_w_was_imputed,f.ac_power_w_was_imputed,
                   f.water_heater_power_w_was_imputed,f.washing_machine_power_w_was_imputed,
                   g.is_dr_event
            FROM read_parquet({h}) h
            JOIN read_parquet({c}) f USING(timestamp,client_id)
            JOIN read_csv_auto('{grid}',header=true) g USING(timestamp)
            WHERE h.client_id=? AND h.timestamp>=?::TIMESTAMP AND h.timestamp<?::TIMESTAMP
            ORDER BY h.timestamp
        """
        with self.connect() as con:
            arr = con.execute(query, [client_id, start, end]).fetchnumpy()
        # DuckDB uses masked arrays for nullable parquet columns.  Preserve null
        # submeter readings as NaN; they mean "unavailable", never zero.
        return {
            k: np.asarray(np.ma.filled(v, np.nan) if np.ma.isMaskedArray(v) else v)
            for k, v in arr.items()
        }


def frozen_mask(values: np.ndarray, minimum_run: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=bool)
    if len(values) == 0:
        return out
    change = np.r_[True, values[1:] != values[:-1], True]
    edges = np.flatnonzero(change)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= minimum_run:
            out[a:b] = True
    return out


def fit_transforms(repo: NILMRepository, cfg: dict, fit_households: Sequence[str]) -> tuple[dict, dict]:
    qcfg = cfg["quality"]
    bounds = cfg["splits"]
    thresholds = np.asarray(qcfg["activity_thresholds_w"], dtype=np.float64)
    aggregate_parts, diff_parts = [], []
    target_parts = [[] for _ in TARGET_COLUMNS]
    activity_counts = np.zeros(3, dtype=np.int64)
    valid_counts = np.zeros(3, dtype=np.int64)
    excesses = []
    source_counts = {"train_rows_scanned": 0, "fit_households": list(fit_households)}

    raw_by_household = []
    for client in fit_households:
        raw = repo.load_raw_partition(client, "train", bounds)
        raw_by_household.append(raw)
        agg = raw["aggregate_power_w"].astype(np.float64)
        good = np.isfinite(agg) & (raw["aggregate_power_w_was_imputed"] == 0)
        aggregate_parts.append(agg[good])
        d = np.diff(agg)
        dg = good[1:] & good[:-1]
        diff_parts.append(d[dg])
        source_counts["train_rows_scanned"] += len(agg)

    all_aggregate = np.concatenate(aggregate_parts)
    aggregate_cap = float(np.quantile(all_aggregate, qcfg["aggregate_spike_quantile"]))
    clipped_good = all_aggregate[all_aggregate <= aggregate_cap]
    agg_mean = float(clipped_good.mean())
    agg_std = float(max(clipped_good.std(), 1.0))
    all_diff = np.concatenate(diff_parts)
    diff_cap = float(np.quantile(np.abs(all_diff), qcfg["difference_abs_quantile"]))
    diff_clean = all_diff[np.abs(all_diff) <= diff_cap]
    diff_std = float(max(diff_clean.std(), 1.0))

    target_caps, power_scales = [], []
    for j, col in enumerate(TARGET_COLUMNS):
        for raw in raw_by_household:
            y = raw[col].astype(np.float64)
            good = np.isfinite(y) & (raw[f"{col}_was_imputed"] == 0)
            target_parts[j].append(y[good])
        values = np.concatenate(target_parts[j])
        empirical_cap = float(np.quantile(values, qcfg["target_spike_quantile"]))
        cap = min(empirical_cap, float(qcfg["target_semantic_caps_w"][j]))
        kept = values[values <= cap]
        scale = float(max(np.quantile(kept, qcfg["power_scale_quantile"]), thresholds[j], 1.0))
        target_caps.append(cap)
        power_scales.append(scale)

    for raw in raw_by_household:
        agg = raw["aggregate_power_w"].astype(np.float64)
        ys = np.column_stack([raw[x].astype(np.float64) for x in TARGET_COLUMNS])
        imp = np.column_stack([raw[f"{x}_was_imputed"].astype(bool) for x in TARGET_COLUMNS])
        valid = np.isfinite(ys) & ~imp & (ys <= np.asarray(target_caps))
        valid_counts += valid.sum(axis=0)
        activity_counts += ((ys >= thresholds) & valid).sum(axis=0)
        all_valid = valid.all(axis=1) & np.isfinite(agg) & (raw["aggregate_power_w_was_imputed"] == 0)
        excesses.append(np.maximum(ys[all_valid].sum(axis=1) - agg[all_valid], 0.0))
    negatives = valid_counts - activity_counts
    pos_weight = np.clip(negatives / np.maximum(activity_counts, 1), 1.0, qcfg["max_positive_class_weight"])
    excess = np.concatenate(excesses)
    positive_excess = excess[excess > 0]
    tolerance = float(np.quantile(positive_excess, qcfg["consistency_tolerance_quantile"])) if len(positive_excess) else 0.0

    transforms = {
        "appliance_order": list(APPLIANCES), "target_columns": list(TARGET_COLUMNS), "units": "watts",
        "activity_thresholds_w": thresholds.tolist(), "activity_label_policy": "pointwise power >= threshold",
        "aggregate_mean_w": agg_mean, "aggregate_std_w": agg_std, "aggregate_spike_cap_w": aggregate_cap,
        "difference_std_w": diff_std, "difference_abs_cap_w": diff_cap,
        "target_caps_w": target_caps, "appliance_power_scales_w": power_scales,
        "activity_positive_class_weights": pos_weight.tolist(),
        "consistency_tolerance_w": tolerance,
        "fitted_split": "train", "fitted_households": list(fit_households),
    }
    source_counts.update({
        "valid_target_counts": valid_counts.tolist(), "active_target_counts": activity_counts.tolist(),
        "positive_excess_count": int(len(positive_excess)),
    })
    return transforms, source_counts


def prepare_partition(repo: NILMRepository, cfg: dict, transforms: dict, client_id: str, split: str) -> PartitionData:
    raw = repo.load_raw_partition(client_id, split, cfg["splits"])
    ts = raw["timestamp"].astype("datetime64[us]")
    agg = raw["aggregate_power_w"].astype(np.float32)
    targets = np.column_stack([raw[x].astype(np.float32) for x in TARGET_COLUMNS])
    agg_imp = raw["aggregate_power_w_was_imputed"].astype(bool)
    target_imp = np.column_stack([raw[f"{x}_was_imputed"].astype(bool) for x in TARGET_COLUMNS])
    frozen = frozen_mask(agg, int(cfg["quality"]["frozen_run_minutes"]))
    spike = agg > float(transforms["aggregate_spike_cap_w"])
    gaps = np.zeros(len(ts), dtype=bool)
    if len(ts) > 1:
        gaps[1:] = np.diff(ts).astype("timedelta64[s]").astype(np.int64) != 60
    nonfinite = ~np.isfinite(agg)
    bad = nonfinite | agg_imp | frozen | spike | gaps
    target_caps = np.asarray(transforms["target_caps_w"], dtype=np.float32)
    contaminated = target_imp | ~np.isfinite(targets) | (targets > target_caps)
    target_valid = ~contaminated
    thresholds = np.asarray(transforms["activity_thresholds_w"], dtype=np.float32)
    activity = (targets >= thresholds).astype(np.float32)
    clipped = np.minimum(agg, float(transforms["aggregate_spike_cap_w"]))
    norm = (clipped - float(transforms["aggregate_mean_w"])) / float(transforms["aggregate_std_w"])
    diffs = np.zeros_like(agg)
    if len(agg) > 1:
        diffs[1:] = np.diff(clipped)
    diffs = np.clip(diffs, -float(transforms["difference_abs_cap_w"]), float(transforms["difference_abs_cap_w"]))
    diffs /= float(transforms["difference_std_w"])
    inputs = np.stack([norm, diffs]).astype(np.float32)
    return PartitionData(client_id, split, ts, agg, targets, agg_imp, target_imp,
                         raw["is_dr_event"].astype(bool), bad, target_valid, inputs, activity,
                         frozen, spike, contaminated, gaps, nonfinite)


def endpoint_quality(part: PartitionData, context: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row validity and a stable reason for invalid input context."""
    n = len(part.timestamps)
    valid = np.zeros(n, dtype=bool)
    reasons = np.full(n, "insufficient_historical_context", dtype=object)
    if n < context:
        return valid, reasons
    checks = (
        (part.aggregate_nonfinite | part.aggregate_imputed, "missing_or_imputed_input"),
        (part.timestamp_gap, "invalid_timestamp_continuity"),
        (part.aggregate_spike, "aggregate_spike"),
        (part.frozen, "frozen_reading"),
    )
    ends = np.arange(context - 1, n, dtype=np.int64)
    reason_assigned = np.zeros(len(ends), dtype=bool)
    for mask, label in checks:
        cs = np.r_[0, np.cumsum(mask.astype(np.int64))]
        affected = (cs[ends + 1] - cs[ends - context + 1]) > 0
        choose = affected & ~reason_assigned
        reasons[ends[choose]] = label
        reason_assigned |= affected
    valid_ends = ends[~reason_assigned]
    valid[valid_ends] = True
    reasons[valid_ends] = "ok"
    return valid, reasons


def valid_endpoints(part: PartitionData, context: int) -> np.ndarray:
    if len(part.timestamps) < context:
        return np.empty(0, dtype=np.int64)
    bad = part.bad_context_point.astype(np.int64)
    cs = np.r_[0, np.cumsum(bad)]
    ends = np.arange(context - 1, len(bad), dtype=np.int64)
    starts = ends - context + 1
    context_bad = cs[ends + 1] - cs[starts]
    # Targets are never required for operational inference.  Training/evaluation
    # masks targets independently, while aggregate-only homes remain inferable.
    return ends[context_bad == 0]


class WindowDataset(Dataset):
    def __init__(self, partitions: Sequence[PartitionData], context: int = 256):
        self.partitions = list(partitions)
        self.context = context
        pi, ei = [], []
        for i, part in enumerate(self.partitions):
            endpoints = valid_endpoints(part, context)
            pi.append(np.full(len(endpoints), i, dtype=np.int16))
            ei.append(endpoints)
        self.partition_index = np.concatenate(pi) if pi else np.empty(0, dtype=np.int16)
        self.endpoint_index = np.concatenate(ei) if ei else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.endpoint_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        part = self.partitions[int(self.partition_index[index])]
        end = int(self.endpoint_index[index])
        start = end - self.context + 1
        return {
            "inputs": torch.from_numpy(part.inputs[:, start:end + 1]),
            "power_w": torch.from_numpy(part.targets_w[end]),
            "activity": torch.from_numpy(part.activity[end]),
            "target_valid": torch.from_numpy(part.target_valid[end]),
            "aggregate_w": torch.tensor(part.aggregate_w[end], dtype=torch.float32),
        }

    def sampling_weights(self, active_weight: float, transition_weight: float) -> torch.Tensor:
        out = np.ones(len(self), dtype=np.float64)
        for i, (pidx, end) in enumerate(zip(self.partition_index, self.endpoint_index)):
            part = self.partitions[int(pidx)]
            active = bool((part.activity[end] * part.target_valid[end]).any())
            transition = end > 0 and bool(((part.activity[end] != part.activity[end - 1]) & part.target_valid[end]).any())
            out[i] = transition_weight if transition else (active_weight if active else 1.0)
        return torch.from_numpy(out)


def quality_summary(partitions: Iterable[PartitionData], context: int) -> dict:
    result = {"rows": 0, "aggregate_imputed": 0, "aggregate_spikes": 0, "frozen_points": 0,
              "target_imputed": [0, 0, 0], "target_contaminated": [0, 0, 0],
              "valid_windows": 0, "candidate_windows": 0}
    for p in partitions:
        result["rows"] += len(p.timestamps)
        result["aggregate_imputed"] += int(p.aggregate_imputed.sum())
        result["aggregate_spikes"] += int(p.aggregate_spike.sum())
        result["frozen_points"] += int(p.frozen.sum())
        result["target_imputed"] = (np.asarray(result["target_imputed"]) + p.target_imputed.sum(axis=0)).tolist()
        result["target_contaminated"] = (np.asarray(result["target_contaminated"]) + p.target_contaminated.sum(axis=0)).tolist()
        result["candidate_windows"] += max(0, len(p.timestamps) - context + 1)
        result["valid_windows"] += len(valid_endpoints(p, context))
    return result
