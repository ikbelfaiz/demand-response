import streamlit as st

from backend.services.analytics_service import display_frequency
from backend.services.regional_service import (regional_daily_trends, regional_peak_periods, regional_summary,
    regional_timeseries, reported_comparison_summary)
from frontend import charts
from frontend.components import page_header, render_chart, section


def render(data) -> None:
    page_header("Regional scope • MW", "Regional Grid Analysis", "TUN demand and reported production indicators for grid-aware research.")
    summary=regional_summary(data); cols=st.columns(4)
    cols[0].metric("Average demand",f"{summary['zone_demand_mw_average']:.1f} MW")
    cols[1].metric("Peak demand",f"{summary['zone_demand_mw_peak']:.1f} MW")
    cols[2].metric("Average STEG production",f"{summary['system_production_mw_average']:.1f} MW")
    cols[3].metric("Average regional PV",f"{summary['zone_pv_production_mw_average']:.1f} MW")
    frequency=display_frequency(data); series=regional_timeseries(data,frequency)
    section("Regional Electricity Demand",f"Displayed with {frequency} mean aggregation.")
    render_chart(charts.regional_demand(series))
    section("Reported Production Versus Demand","STEG and PV are the production channels supplied by this dataset. Imports, storage, losses and other sources may be absent.")
    render_chart(charts.production_demand(series))
    section("Regional PV Production")
    render_chart(charts.pv_production(series))
    section("Daily Regional Trends")
    render_chart(charts.regional_daily(regional_daily_trends(data)))
    left,right=st.columns([1,1])
    with left:
        section("Highest Regional Demand Readings","High demand is not automatically classified as an emergency or DR decision.")
        peaks=regional_peak_periods(data); peaks["timestamp"]=peaks.timestamp.dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(peaks.rename(columns={"timestamp":"Time","zone_demand_mw":"Demand (MW)"}),hide_index=True,width="stretch")
    with right:
        section("Reported-Signal Comparison")
        comparison=reported_comparison_summary(data)["mean_reported_production_minus_demand_mw"]
        st.metric("Mean reported production − demand",f"{comparison:.1f} MW" if comparison is not None else "—")
        st.caption("This arithmetic comparison is not a deficit, adequacy or emergency assessment because the dataset may omit other balancing components.")
