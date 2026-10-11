"""Device recommendation for the event window: which appliance the chatbot asks the household to act on.

Two sources, chosen by default_recommender():

  NILMRecommender      the team's TCN NILM model (branch NILM: backend/services/nilm_service.py,
                       checkpoint nilm_research/outputs/executed_cpu/tcn/best.pt). It disaggregates the
                       smart-meter signal of the SAME clock window on the last LOOKBACK_DAYS days before
                       the decision time, and recommends the appliance that used the most energy there.
  DeclaredDeviceRules  fallback while the NILM model is not installed: transparent rules on the devices
                       declared in households_info.csv and the temperature.

Output: Recommendation(device, action, device_kwh, confidence, source)   (device None = no recommendation)
device_kwh is the appliance's usual energy in the window; pricing.expected_saving applies the share of it
the action saves. Devices the household refused are always skipped. The system never controls appliances.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ACTION_FOR_DEVICE = {"ac": "reduce_ac", "water_heater": "shift_water_heater",
                     "washing_machine": "shift_washing_machine"}


@dataclass
class Recommendation:
    device: str | None
    action: str | None
    device_kwh: float
    confidence: float
    source: str = "declared-device rules"


NONE = Recommendation(None, None, 0.0, 0.0)


class DeclaredDeviceRules:
    """Rules, as a share of the household's usual kWh in the window:
      hot window (≥ 27 °C) and AC         → reduce_ac,             45%
      electric water heater               → shift_water_heater,    35% (max 1.5 kWh per hour of window)
      AC used for heating (≤ 14 °C)       → reduce_ac,             30%
      washing machine                     → shift_washing_machine, 10%
    """
    source = "declared-device rules"

    def recommend(self, household, usual_window_kwh: float, window_hours: float, mean_temp_c: float,
                  refused: set[str], **_) -> Recommendation:
        cands = []
        if household["has_ac"] and mean_temp_c >= 27:
            cands.append(("ac", "reduce_ac", 0.45 * usual_window_kwh, 0.6))
        if household["has_electric_water_heater"]:
            cands.append(("water_heater", "shift_water_heater",
                          min(0.35 * usual_window_kwh, 1.5 * window_hours), 0.5))
        if household["has_ac"] and mean_temp_c <= 14:
            cands.append(("ac", "reduce_ac", 0.30 * usual_window_kwh, 0.45))
        if household["has_washing_machine"]:
            cands.append(("washing_machine", "shift_washing_machine", 0.10 * usual_window_kwh, 0.3))
        for device, action, kwh, conf in cands:
            if device not in refused and kwh > 0.05:
                return Recommendation(device, action, round(kwh, 3), conf, self.source)
        return NONE


class NILMRecommender:
    """Recommendation from the team's NILM model.

    For each of the last LOOKBACK_DAYS days whose window ended before `asof`, the model estimates the
    energy of each appliance in the window (valid minutes only, no extrapolation). The appliance with the
    largest mean energy is recommended. Confidence = share of those days on which it really ran in the
    window (≥ ACTIVE_KWH) × data coverage. The NILM activity probabilities are not used as confidence
    (not calibrated, see NILM_INTEGRATION.md). Washing-machine estimates are experimental, so their
    confidence is halved. Falls back to the rules if the model cannot answer for this household.
    """
    LOOKBACK_DAYS = 3
    ACTIVE_KWH = 0.10
    MIN_COVERAGE = 0.5
    EXPERIMENTAL = {"washing_machine": 0.5}

    def __init__(self, predict=None, fallback=None, version: str = "nilm-tcn"):
        if predict is None:
            from backend.services.nilm_service import predict_appliances as predict
        self._predict = predict
        self.fallback = fallback or DeclaredDeviceRules()
        self.source = version

    def window_energy(self, client_id: str, start: pd.Timestamp, end: pd.Timestamp,
                      asof: pd.Timestamp) -> pd.DataFrame:
        """kWh per appliance in the same clock window on the last days (rows = days)."""
        rows, d = [], start.normalize() - pd.Timedelta(days=1)
        offset, length = start - start.normalize(), end - start
        while len(rows) < self.LOOKBACK_DAYS and d >= start.normalize() - pd.Timedelta(days=10):
            s = d + offset
            if s + length <= asof:
                try:
                    res = self._predict(client_id, s, s + length)
                except ValueError:                       # outside the dataset, unknown household …
                    break
                summ = res.summary
                row = {"day": d, "coverage": float(summ.get("coverage_fraction", 0.0))}
                for dev, info in summ["appliances"].items():
                    row[dev] = info.get("observed_estimated_energy_kwh")
                rows.append(row)
            d -= pd.Timedelta(days=1)
        return pd.DataFrame(rows)

    def recommend(self, household, usual_window_kwh: float, window_hours: float, mean_temp_c: float,
                  refused: set[str], *, client_id: str | None = None, start=None, end=None, asof=None,
                  **_) -> Recommendation:
        client_id = client_id or household.name
        if start is None or end is None or asof is None:
            return self.fallback.recommend(household, usual_window_kwh, window_hours, mean_temp_c, refused)
        try:
            days = self.window_energy(str(client_id), pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(asof))
        except Exception:                                # model missing, CUDA error … → never block an event
            days = pd.DataFrame()
        if days.empty or days["coverage"].mean() < self.MIN_COVERAGE:
            return self.fallback.recommend(household, usual_window_kwh, window_hours, mean_temp_c, refused)
        coverage = float(days["coverage"].mean())
        best = None
        for dev, action in ACTION_FOR_DEVICE.items():
            if dev in refused or dev not in days:
                continue
            kwh = pd.to_numeric(days[dev], errors="coerce")
            if kwh.notna().sum() == 0:
                continue
            mean_kwh = float(kwh.mean())
            conf = float((kwh >= self.ACTIVE_KWH).mean()) * coverage * self.EXPERIMENTAL.get(dev, 1.0)
            if mean_kwh >= self.ACTIVE_KWH and (best is None or mean_kwh > best.device_kwh):
                best = Recommendation(dev, action, round(mean_kwh, 3), round(conf, 2), self.source)
        return best or Recommendation(None, None, 0.0, 0.0, self.source)


def nilm_installed() -> bool:
    try:
        from backend.services.nilm_service import model_status
    except Exception:            # NILM branch not merged, torch missing …
        return False
    ok, _ = model_status()
    return bool(ok)


def default_recommender():
    """The team's NILM model when it is installed (code + checkpoint), otherwise the declared-device rules."""
    if nilm_installed():
        from backend.services.nilm_service import model_status
        meta = model_status()[1]
        version = f"nilm-{meta.get('model_kind', 'tcn')}" if isinstance(meta, dict) else "nilm-tcn"
        return NILMRecommender(version=version)
    return DeclaredDeviceRules()
