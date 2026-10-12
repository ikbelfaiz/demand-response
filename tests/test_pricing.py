"""Pricing engine (section 7.2): critical peak pricing on the STEG low-voltage tariff."""
import pytest

from backend.intelligence import pricing as p

B218 = p.band_price(250, 3)


def test_band_thresholds_uniform_billing():
    assert p.band_price(200, 3).price == 176
    assert p.band_price(300, 3).price == 218
    assert p.band_price(301, 3).price == 341
    assert p.band_price(500, 3).price == 341
    assert p.band_price(501, 3).price == 414


def test_economic_band_needs_small_contract():
    assert p.band_price(45, 1) == p.Band(62, True)
    assert p.band_price(80, 2) == p.Band(96, True)
    assert p.band_price(80, 3) == p.Band(176, False)      # > 2 kVA: normal band from the first kWh
    assert p.band_price(120, 2) == p.Band(176, False)     # > 100 kWh: normal band


@pytest.mark.parametrize("level,peak", [(1, 283), (2, 327), (3, 436)])
def test_peak_price_eq6(level, peak):
    assert p.peak_price(B218, level) == peak


def test_peak_price_table():
    expected = {176: (229, 264, 352), 218: (283, 327, 436), 341: (443, 512, 682), 414: (538, 621, 828)}
    for kwh, price in [(150, 176), (250, 218), (400, 341), (600, 414)]:
        band = p.band_price(kwh, 3)
        assert tuple(p.peak_price(band, lvl) for lvl in (1, 2, 3)) == expected[price]


@pytest.mark.parametrize("kwh,cost,surcharge", [(3.2, 1.046, 0.349), (2.75, 0.899, 0.300), (1.7, 0.556, 0.185)])
def test_worked_example_7_2(kwh, cost, surcharge):
    assert p.window_cost(kwh, B218, 2) == cost
    assert p.event_surcharge(kwh, B218, 2) == surcharge


def test_exemptions():
    assert p.event_surcharge(3.2, p.band_price(80, 2), 3) == 0.0          # economic band
    assert p.event_surcharge(3.2, B218, 2, cancelled=True) == 0.0         # cancelled event
    assert p.peak_price(p.band_price(80, 2), 2) == 96


def test_expected_saving_notification_card():
    # 1.5 kWh of AC, reduced by 30% → 0.45 kWh avoided at 327 millimes ≈ 0.15 DT
    assert p.expected_saving(1.5, "reduce_ac", B218, 2) == 0.147
    # shifted water heater: pays the normal price later → only the surcharge part is saved
    assert p.expected_saving(1.5, "shift_water_heater", B218, 2) == round(1.5 * 109 / 1000, 3)


def test_monthly_bill_and_cap():
    res = p.monthly_bill(250, 3, [{"window_kwh": k} for k in (3.2, 2.75, 2.8, 3.0)])
    assert res["energy_dt"] == 54.5
    assert res["dr_surcharge_dt"] == 1.281
    assert res["power_fee_dt"] == 2.1
    assert res["total_before_tax_dt"] == 57.881
    capped = p.monthly_bill(150, 3, [{"window_kwh": 30, "level": 3}])  # 30 kWh at +100% → over 10% of energy
    assert capped["cap_applied"] and capped["dr_surcharge_dt"] == round(0.10 * 150 * 176 / 1000, 3)
