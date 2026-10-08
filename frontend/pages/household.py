import streamlit as st

from backend.services.analytics_service import chart_series, display_frequency
from backend.services.energy_service import energy_summary
from backend.services.household_service import (appliance_breakdown, appliance_columns, appliance_label,
    daily_trend, hourly_heatmap, hourly_profile, household_distribution, peak_periods)
from frontend import charts
from frontend.components import page_header, render_chart, section


def render(data) -> None:
    page_header("Household scope • kW and kWh", "Household Analysis", "Detailed load behavior for client C001 using only measured household channels.")
    summary=energy_summary(data); cols=st.columns(4)
    cols[0].metric("Average demand",f"{summary['average_kw']:.2f} kW")
    cols[1].metric("Peak demand",f"{summary['peak_kw']:.2f} kW")
    cols[2].metric("Load factor",f"{summary['load_factor']:.1%}")
    cols[3].metric("Original load coverage",f"{summary['coverage']:.2%}",help=f"Bounded imputed samples: {summary['imputed_samples']:,}")
    plotted=chart_series(data,display_frequency(data))
    section("Household Load Curve","Gaps that remain after bounded cleaning are rendered as unavailable observations.")
    render_chart(charts.household_load(plotted))
    left,right=st.columns(2)
    with left:
        section("Hourly Consumption Pattern")
        render_chart(charts.hourly_pattern(hourly_profile(data)))
    with right:
        section("Demand Distribution")
        render_chart(charts.demand_distribution(household_distribution(data)))
    section("Hourly Consumption Heatmap","Mean valid household power for each date and hour.")
    render_chart(charts.load_heatmap(hourly_heatmap(data)))
    section("Measured Appliance Demand","Available channels are discovered from the standardized dataset mapping; they do not form a complete end-use balance.")
    available=appliance_columns(data); labels={column:appliance_label(column) for column in available}
    render_chart(charts.appliance_stack(plotted,labels))
    render_chart(charts.appliance_energy(appliance_breakdown(data)))
    left,right=st.columns(2)
    with left:
        section("Daily Integrated Energy","Hover for original measurement coverage; unresolved gaps are not fabricated.")
        render_chart(charts.daily_energy(daily_trend(data)))
    with right:
        section("Highest Household Readings","Peak observations are descriptive household measurements, not DR activation signals.")
        peaks=peak_periods(data); peaks["timestamp"]=peaks.timestamp.dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(peaks.rename(columns={"timestamp":"Time","household_power_kw":"Power (kW)"}),hide_index=True,width="stretch")
