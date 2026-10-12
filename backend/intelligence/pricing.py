"""Pricing engine — section 7.2 (critical peak pricing on top of the STEG low-voltage tariff).

All prices in millimes/kWh, all amounts in DT, before tax.
  Peak price            = P × (1 + s)                       (6)
  Event surcharge (DT)  = E_window × P × s ÷ 1000            (7)
"""
from dataclasses import dataclass

# STEG residential low-voltage bands: (upper limit kWh/month, price). Whole month billed at the band reached.
ECONOMIC_BANDS = [(50, 62), (100, 96)]                       # only for 1–2 kVA contracts and ≤ 100 kWh/month
NORMAL_BANDS = [(200, 176), (300, 218), (500, 341), (float("inf"), 414)]
LEVELS = {1: 0.30, 2: 0.50, 3: 1.00}                         # surcharge s per DR level
POWER_FEE_DT_PER_KVA = 0.700                                 # per month, unchanged by DR
MONTHLY_CAP_SHARE = 0.10                                     # DR surcharges ≤ 10% of the month's energy cost

# Recommended actions: (reduction rate of the device's kWh, energy used later at the normal price?)
ACTIONS = {
    "reduce_ac": (0.30, False),          # starting value, to recalibrate on measured results
    "switch_off_ac": (1.00, False),
    "shift_water_heater": (1.00, True),
    "shift_washing_machine": (1.00, True),
}


@dataclass(frozen=True)
class Band:
    price: int          # P, millimes/kWh
    economic: bool      # economic band → exempt from the DR surcharge


def band_price(monthly_kwh: float, kva: float) -> Band:
    if kva <= 2 and monthly_kwh <= 100:
        for limit, price in ECONOMIC_BANDS:
            if monthly_kwh <= limit:
                return Band(price, True)
    for limit, price in NORMAL_BANDS:
        if monthly_kwh <= limit:
            return Band(price, False)


def surcharge_rate(band: Band, level: int, exempt: bool = False) -> float:
    return 0.0 if (band.economic or exempt) else LEVELS[level]


def peak_price(band: Band, level: int) -> int:
    """Equation (6). Economic band: unchanged price."""
    return round(band.price * (1 + surcharge_rate(band, level)))


def expected_saving(device_kwh: float, action: str, band: Band, level: int) -> float:
    """Notification estimate: ΔE × (peak price − price paid later) ÷ 1000."""
    if action not in ACTIONS:
        return 0.0
    rate, shifted = ACTIONS[action]
    delta = device_kwh * rate
    later = band.price if shifted else 0
    return round(delta * (band.price * (1 + surcharge_rate(band, level)) - later) / 1000, 3)


def projected_extra_cost(kwh_so_far: float, usual_remaining_kwh: float, band: Band, level: int) -> float:
    """Reminder: extra cost if consumption continues at its usual pace."""
    return round((kwh_so_far + usual_remaining_kwh) * band.price * surcharge_rate(band, level) / 1000, 3)


def event_surcharge(window_kwh: float, band: Band, level: int, cancelled: bool = False) -> float:
    """Equation (7) — real surcharge, from meter readings (missing slots are excluded upstream)."""
    if cancelled:
        return 0.0
    return round(window_kwh * band.price * surcharge_rate(band, level) / 1000, 3)


def window_cost(window_kwh: float, band: Band, level: int, cancelled: bool = False) -> float:
    s = 0.0 if cancelled else surcharge_rate(band, level)
    return round(window_kwh * band.price * (1 + s) / 1000, 3)


def monthly_bill(monthly_kwh: float, kva: float, events: list[dict]) -> dict:
    """events: [{"window_kwh": float, "level": int, "cancelled": bool}] — uses the REAL band of the month."""
    band = band_price(monthly_kwh, kva)
    energy = monthly_kwh * band.price / 1000
    dr_raw = sum(event_surcharge(e["window_kwh"], band, e.get("level", 2), e.get("cancelled", False)) for e in events)
    cap = MONTHLY_CAP_SHARE * energy
    dr = min(dr_raw, cap)
    fee = kva * POWER_FEE_DT_PER_KVA
    return {"band_price": band.price, "economic": band.economic,
            "energy_dt": round(energy, 3), "dr_surcharge_raw_dt": round(dr_raw, 3),
            "dr_surcharge_dt": round(dr, 3), "cap_applied": dr_raw > cap,
            "power_fee_dt": round(fee, 3), "total_before_tax_dt": round(energy + dr + fee, 3)}
