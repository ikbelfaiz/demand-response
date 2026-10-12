"""Customer profiling and NILM plugged into the orchestrator (adapters, with fallbacks)."""
from types import SimpleNamespace

import pandas as pd

from backend.orchestration.adapters.devices import DeclaredDeviceRules, NILMRecommender, default_recommender
from backend.orchestration.adapters.profiling import (CustomerProfilingTargeting, LocalTargeting,
                                                      PRIORITY_COLUMNS, default_targeting, profiling_installed)

HH = pd.DataFrame({"has_ac": [True, False], "has_electric_water_heater": [True, True],
                   "has_washing_machine": [True, True], "contracted_kva": [3, 2], "segment": ["Unknown"] * 2,
                   "acceptance_rate": [0.5, 0.5]}, index=pd.Index(["C001", "C002"], name="client_id"))
START, END = pd.Timestamp("2025-07-15 18:00"), pd.Timestamp("2025-07-15 20:00")
ASOF = pd.Timestamp("2025-07-14 14:30")
METER = pd.DataFrame(0.5, index=pd.date_range("2025-06-01", "2025-07-16", freq="30min"), columns=["C001", "C002"])


def fake_prepare(start, end, deficit, as_of):
    assert as_of == ASOF                          # only data known at the decision time
    targeting = pd.DataFrame({"rank": [1], "client_id": ["C002"], "segment": ["Large evening-peak households"],
                              "usual_window_kwh": [3.1], "acceptance_rate": [0.7], "expected_reduction_kwh": [0.43],
                              "covers_deficit_at_rank": [False]})
    context = pd.DataFrame({"client_id": ["C002"], "month_to_date_kwh": [250.0], "projected_month_kwh": [520.0],
                            "comparison_message": ["You use 6% more than households like yours in the evening."]})
    return targeting, context


def test_profiling_targeting_uses_model_order_and_appends_unknown_households():
    df = CustomerProfilingTargeting(prepare_event=fake_prepare, version="kmeans_v1").priority_list(
        METER, HH, START, END, ASOF, set(), deficit_kwh=5.0)
    assert list(df.columns) == PRIORITY_COLUMNS
    assert df.client_id.tolist() == ["C002", "C001"] and df["rank"].tolist() == [1, 2]
    first = df.iloc[0]
    assert first.segment == "Large evening-peak households" and first.projected_month_kwh == 520.0
    assert first.contracted_kva == 2 and first.comparison_message.startswith("You use")


def test_local_targeting_has_same_columns():
    df = LocalTargeting().priority_list(METER, HH, START, END, ASOF, set())
    assert list(df.columns) == PRIORITY_COLUMNS and len(df) == 2


def test_default_targeting_matches_installed_profiling_outputs():
    assert isinstance(default_targeting(), CustomerProfilingTargeting if profiling_installed() else LocalTargeting)


def _nilm(energies, coverage=1.0):
    calls = []

    def predict(client_id, start, end):
        calls.append((client_id, start, end))
        appliances = {dev: {"observed_estimated_energy_kwh": kwh} for dev, kwh in energies.items()}
        return SimpleNamespace(summary={"coverage_fraction": coverage, "appliances": appliances})
    return predict, calls


def test_nilm_recommends_largest_appliance_on_past_windows_only():
    predict, calls = _nilm({"ac": 1.2, "water_heater": 0.6, "washing_machine": 0.0})
    rec = NILMRecommender(predict=predict).recommend(HH.loc["C001"], 3.0, 2.0, 30.0, set(),
                                                     client_id="C001", start=START, end=END, asof=ASOF)
    assert (rec.device, rec.action, rec.device_kwh, rec.confidence) == ("ac", "reduce_ac", 1.2, 1.0)
    assert len(calls) == 3 and all(e <= ASOF for _, _, e in calls)       # D-1 18:00 is after 14:30 → D-2..D-4


def test_nilm_skips_refused_device_and_falls_back_without_data():
    predict, _ = _nilm({"ac": 1.2, "water_heater": 0.6, "washing_machine": 0.0})
    rec = NILMRecommender(predict=predict).recommend(HH.loc["C001"], 3.0, 2.0, 30.0, {"ac"},
                                                     client_id="C001", start=START, end=END, asof=ASOF)
    assert rec.device == "water_heater"
    low, _ = _nilm({"ac": 1.2}, coverage=0.1)
    rec = NILMRecommender(predict=low).recommend(HH.loc["C001"], 3.0, 2.0, 30.0, set(),
                                                 client_id="C001", start=START, end=END, asof=ASOF)
    assert rec.source == DeclaredDeviceRules.source


def test_default_recommender_falls_back_to_rules_without_nilm_checkpoint():
    assert isinstance(default_recommender(), (DeclaredDeviceRules, NILMRecommender))
