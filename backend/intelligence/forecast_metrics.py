"""Forecast metrics with explicit household power and interval-energy semantics."""
from __future__ import annotations

import numpy as np


def forecast_metrics(actual_kw, predicted_kw, *, interval_hours: float = 0.5) -> dict[str, float]:
    actual = np.asarray(actual_kw, dtype=float)
    predicted = np.asarray(predicted_kw, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted arrays must have the same non-empty shape.")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if actual.size == 0:
        raise ValueError("No finite forecast pairs are available.")
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    actual_peak, predicted_peak = int(np.argmax(actual)), int(np.argmax(predicted))
    actual_energy = float(actual.sum() * interval_hours)
    predicted_energy = float(predicted.sum() * interval_hours)
    energy_bias = predicted_energy - actual_energy
    peak_magnitude_bias = float(predicted[predicted_peak] - actual[actual_peak])
    peak_timing_bias_minutes = float((predicted_peak - actual_peak) * interval_hours * 60)
    return {
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error ** 2))),
        "wape_pct": float(np.abs(error).sum() / denominator * 100) if denominator > 0 else float("nan"),
        "daily_energy_error_kwh": energy_bias,
        "daily_energy_absolute_error_kwh": abs(energy_bias),
        "daily_energy_error_pct": (energy_bias / actual_energy * 100)
        if actual_energy != 0 else float("nan"),
        "daily_energy_absolute_error_pct": (abs(energy_bias) / abs(actual_energy) * 100)
        if actual_energy != 0 else float("nan"),
        "peak_magnitude_error_kw": abs(peak_magnitude_bias),
        "peak_magnitude_bias_kw": peak_magnitude_bias,
        "peak_timing_error_minutes": abs(peak_timing_bias_minutes),
        "peak_timing_bias_minutes": peak_timing_bias_minutes,
    }


def forecast_kpis(predicted_kw, *, interval_hours: float = 0.5) -> dict[str, float | int]:
    values = np.asarray(predicted_kw, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite and non-empty.")
    peak_slot = int(np.argmax(values))
    return {
        "daily_energy_kwh": float(values.sum() * interval_hours),
        "peak_power_kw": float(values[peak_slot]),
        "peak_slot": peak_slot,
        "average_power_kw": float(values.mean()),
    }
