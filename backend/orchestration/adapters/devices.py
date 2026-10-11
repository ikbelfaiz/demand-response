"""Device recommendation for the event window.

backend.intelligence.nilm is not trained yet (estimate_appliances raises NotImplementedError), so the
orchestrator falls back to transparent rules on the devices DECLARED in households_info.csv and the
temperature. This is a recommendation rule, not an appliance estimate; it is replaced automatically
as soon as NILM inference is implemented with the same output.

Output: Recommendation(device, action, device_kwh, confidence)   (device None = no recommendation)
Rules, as a share of the household's usual kWh in the window:
  hot window (≥ 27 °C) and AC         → reduce_ac,             45%
  electric water heater               → shift_water_heater,    35% (max 1.5 kWh per hour of window)
  AC used for heating (≤ 14 °C)       → reduce_ac,             30%
  washing machine                     → shift_washing_machine, 10%
Devices the household refused are skipped.
"""
from dataclasses import dataclass


@dataclass
class Recommendation:
    device: str | None
    action: str | None
    device_kwh: float
    confidence: float
    source: str = "declared-device rules"


class DeclaredDeviceRules:
    def recommend(self, household, usual_window_kwh: float, window_hours: float, mean_temp_c: float,
                  refused: set[str]) -> Recommendation:
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
                return Recommendation(device, action, round(kwh, 3), conf)
        return Recommendation(None, None, 0.0, 0.0)


def default_recommender():
    """NILM when it is implemented, otherwise the declared-device rules."""
    try:
        from backend.intelligence.nilm import estimate_appliances
        estimate_appliances()
    except NotImplementedError:
        return DeclaredDeviceRules()
    except TypeError:
        pass  # NILM implemented (needs arguments): plug its adapter here
    return DeclaredDeviceRules()
