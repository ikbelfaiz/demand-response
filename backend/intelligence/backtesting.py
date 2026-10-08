"""Metrics for interval and event-level historical replay evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(actual, predicted) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=bool); predicted = np.asarray(predicted, dtype=bool)
    if actual.shape != predicted.shape:
        raise ValueError("Classification arrays must have identical shapes.")
    tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum()); tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "true_positive":tp,"true_negative":tn,"false_positive":fp,"false_negative":fn}


def windows_overlap(first_start, first_end, second_start, second_end) -> pd.Timedelta:
    return max(pd.Timedelta(0), min(pd.Timestamp(first_end),pd.Timestamp(second_end)) -
               max(pd.Timestamp(first_start),pd.Timestamp(second_start)))


def event_matching_metrics(predicted: pd.DataFrame, actual: pd.DataFrame, *, replay_days: int,
                           issue_times: dict[pd.Timestamp,pd.Timestamp] | None = None) -> dict[str, float | int]:
    predicted = predicted.copy(); actual = actual.copy()
    pred_match = []
    for pred in predicted.itertuples():
        pred_match.append(any(windows_overlap(pred.start_ts,pred.end_ts,row.start_ts,row.end_ts)>pd.Timedelta(0)
                              for row in actual.itertuples()))
    actual_match, overlaps = [], []
    for row in actual.itertuples():
        intersections = [windows_overlap(row.start_ts,row.end_ts,p.start_ts,p.end_ts) for p in predicted.itertuples()]
        best = max(intersections, default=pd.Timedelta(0)); actual_match.append(best>pd.Timedelta(0))
        overlaps.append(best.total_seconds()/60)
    lead_hours = []
    if issue_times:
        for pred in predicted.itertuples():
            issue = issue_times.get(pd.Timestamp(pred.start_ts).normalize())
            if issue is not None: lead_hours.append((pd.Timestamp(pred.start_ts)-issue).total_seconds()/3600)
    return {
        "predicted_events":len(predicted),"actual_events":len(actual),
        "event_precision":sum(pred_match)/len(pred_match) if pred_match else 0.0,
        "event_recall":sum(actual_match)/len(actual_match) if actual_match else 0.0,
        "false_alerts_per_day":(len(pred_match)-sum(pred_match))/replay_days if replay_days else 0.0,
        "mean_best_overlap_minutes":float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_predicted_lead_hours":float(np.mean(lead_hours)) if lead_hours else 0.0,
    }


def mark_windows(timestamps, windows: pd.DataFrame) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps)); mask = np.zeros(len(index), dtype=bool)
    for row in windows.itertuples():
        mask |= (index >= pd.Timestamp(row.start_ts)) & (index < pd.Timestamp(row.end_ts))
    return mask
