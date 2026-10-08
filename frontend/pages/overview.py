import streamlit as st

from backend.services.analytics_service import chart_series, display_frequency
from backend.services.energy_service import energy_summary
from backend.services.household_service import daily_trend, hourly_profile
from backend.services.regional_service import regional_summary, regional_timeseries
from frontend import charts
from frontend.components import page_header, render_chart, section


def render(data) -> None:
    page_header("Energy monitoring", "Energy Overview", "Household and regional measurements are presented as separate physical scopes.")
    household=energy_summary(data); regional=regional_summary(data)
    cols=st.columns(4)
    energy_label="Household energy" if household["energy_complete"] else "Integrated household energy*"
    cols[0].metric(energy_label,f"{household['energy_kwh']:.2f} kWh",help=f"Original measurement coverage: {household['coverage']:.2%}; bounded imputed samples: {household['imputed_samples']:,}.")
    cols[1].metric("Average household demand",f"{household['average_kw']:.2f} kW")
    cols[2].metric("Peak household demand",f"{household['peak_kw']:.2f} kW")
    cols[3].metric("Household load factor",f"{household['load_factor']:.1%}")
    cols=st.columns(3)
    cols[0].metric("Average regional demand",f"{regional['zone_demand_mw_average']:.1f} MW")
    cols[1].metric("Average STEG production",f"{regional['system_production_mw_average']:.1f} MW")
    cols[2].metric("Average regional PV",f"{regional['zone_pv_production_mw_average']:.1f} MW")
    frequency=display_frequency(data); household_chart=chart_series(data,frequency); regional_chart=regional_timeseries(data,frequency)
    section("Household Consumption",f"C001 aggregate demand • {frequency} mean power for display • analytics remain at one-minute resolution")
    render_chart(charts.household_load(household_chart))
    section("Regional Electricity Demand","TUN regional signal in MW; it is not an aggregation of the displayed household.")
    render_chart(charts.regional_demand(regional_chart))
    section("Regional Production and Demand","All signals are regional MW measurements. The chart does not assert a complete grid balance.")
    render_chart(charts.production_demand(regional_chart))
    section("Regional PV Production")
    render_chart(charts.pv_production(regional_chart))
    left,right=st.columns(2)
    with left:
        section("Household Daily Trend","Integrated energy with source-observation coverage shown on hover; dates with unresolved gaps are partial.")
        render_chart(charts.daily_energy(daily_trend(data)))
    with right:
        section("Household Hourly Pattern","Mean and 95th percentile by hour across the selected period.")
        render_chart(charts.hourly_pattern(hourly_profile(data)))
