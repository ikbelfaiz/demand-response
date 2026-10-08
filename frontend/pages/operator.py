"""Operator perspective pages for the synthetic neighborhood."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data import load_events, load_household, load_household_info, load_participation, operational_household
from backend.services import community_service, event_service, grid_service, household_service
from backend.services.segmentation_service import consumption_quantiles
from frontend.charts import bars, donut, heatmap, lines, style
from frontend.components import page_header, render_chart
from frontend.styles import COLORS


def overview(grid: pd.DataFrame) -> None:
    page_header("Operator / community", "Community overview", "Operational 30-minute view of the 50-household synthetic case study.")
    s = community_service.summary(grid); events = load_events()
    cols = st.columns(4)
    metrics = [("Households", "50"), ("Demand energy", f"{s['energy_kwh']:,.0f} kWh"),
               ("Average demand", f"{s['average_demand_kw']:.1f} kW"), ("Maximum demand", f"{s['maximum_demand_kw']:.1f} kW"),
               ("Available STEG", f"{s['steg_energy_kwh']:,.0f} kWh"), ("Available PV", f"{s['pv_energy_kwh']:,.0f} kWh"),
               ("Average margin", f"{s['average_margin_kw']:.1f} kW"), ("Historical events", str(len(events)))]
    for i, (label, value) in enumerate(metrics): cols[i % 4].metric(label, value)
    st.caption("Supply is a modeled allocation for this synthetic academic case study, not a verified shortage record.")
    render_chart(lines(grid, [("households_consumption_kw","Feeder demand",COLORS["zone"]), ("available_supply_kw","Modeled supply",COLORS["production"])], "Power (kW)"))
    left, right = st.columns(2)
    with left: render_chart(lines(grid, [("steg_production_kw","Allocated STEG",COLORS["production"]),("pv_production_kw","Allocated PV",COLORS["pv"])], "Power (kW)"))
    with right: render_chart(heatmap(community_service.demand_heatmap(grid)))


def community_grid(grid: pd.DataFrame) -> None:
    page_header("Operator / grid", "Community grid", "Modeled supply, demand margin, and historical synthetic peak labels.")
    comparison = grid_service.supply_comparison(grid)
    a,b,c,d = st.columns(4)
    a.metric("Mean demand", f"{comparison.households_consumption_kw.mean():.1f} kW")
    b.metric("Mean supply", f"{comparison.available_supply_kw.mean():.1f} kW")
    c.metric("Minimum margin", f"{comparison.margin_kw.min():.1f} kW")
    d.metric("Peak intervals", f"{int(comparison.is_dr_peak.fillna(0).sum()):,}")
    st.info(grid_service.PEAK_ASSUMPTION)
    render_chart(lines(comparison, [("households_consumption_kw","Demand",COLORS["zone"]),("steg_production_kw","STEG",COLORS["production"]),("pv_production_kw","PV",COLORS["pv"]),("available_supply_kw","Total available",COLORS["event"])], "Power (kW)"))
    freq = community_service.peak_frequency(comparison)
    render_chart(bars(freq, "hour", "peak_intervals", "Historical peak intervals", "30-minute intervals", COLORS["peak"]))


def events(grid: pd.DataFrame) -> None:
    page_header("Operator / history", "Demand Response events", "Recorded synthetic events and household responses; these are not AI predictions.")
    ev, part = load_events(), load_participation(); table = event_service.event_table(ev, part)
    st.metric("Historical events", len(table))
    selected = st.selectbox("Inspect event", table.event_id.tolist(), format_func=lambda x: f"Event {x} — {table.loc[table.event_id.eq(x),'start_ts'].iloc[0]:%d %b %Y %H:%M}")
    row = table.loc[table.event_id.eq(selected)].iloc[0]
    cols=st.columns(6)
    for col,(label,value) in zip(cols,[("Duration",f"{row.duration_minutes:.0f} min"),("Target",f"{row.reduction_target_pct:.0f}%"),("Scenario level",str(row.surcharge_level)),("Accepted",int(row["accept"])),("Declined",int(row["decline"])),("No response",int(row["no_response"]))]): col.metric(label,value)
    window = event_service.event_window(grid, row)
    render_chart(lines(window, [("households_consumption_kw","Demand",COLORS["zone"]),("available_supply_kw","Available supply",COLORS["production"])], "Power (kW)"))
    render_chart(donut(["Accept","Decline","No response"],[row["accept"],row["decline"],row["no_response"]],f"Participation: {row.participation_pct:.1f}% accepted"))
    st.dataframe(part.loc[part.event_id.eq(selected)], hide_index=True, width="stretch", height=280)
    st.warning("Automated DR event forecasting and decision models are not yet implemented. Scenario level is synthetic; no monetary tariff is inferred.")


@st.cache_data(show_spinner="Computing household portfolio metrics…")
def _portfolio(start, end): return household_service.portfolio_summary(start, end)


def households(grid: pd.DataFrame, start, end) -> None:
    page_header("Operator / households", "Household analytics", "Transparent comparisons and rule-based usage groups across all 50 homes.")
    portfolio = _portfolio(start, end); info = load_household_info()
    portfolio["segment"] = consumption_quantiles(portfolio.energy_kwh).astype(str)
    render_chart(bars(portfolio.sort_values("energy_kwh"), "client_id", "energy_kwh", "Energy", "Observed energy (kWh)"))
    left,right=st.columns(2)
    with left:
        fig=go.Figure(go.Histogram(x=portfolio.energy_kwh,nbinsx=15,marker_color=COLORS["household"])); render_chart(style(fig,"Households","Energy (kWh)"))
    with right:
        own={"AC":int(info.has_ac.sum()),"Second AC":int(info.has_second_ac.sum()),"Electric water heater":int(info.has_electric_water_heater.sum()),"Washing machine":int(info.has_washing_machine.sum()),"Daytime occupied":int(info.occupied_daytime.sum()),"Submeter":int(info.has_submeter.sum())}
        render_chart(bars(pd.DataFrame({"attribute":own.keys(),"count":own.values()}),"attribute","count","Households","Count",COLORS["pv"]))
    selected=st.multiselect("Compare load profiles", info.client_id.tolist(), default=info.client_id.tolist()[:2], max_selections=5)
    fig=go.Figure()
    for client in selected:
        raw=load_household(client,start,end); op=operational_household(raw); prof=household_service.hourly_profile(op)
        fig.add_trace(go.Scatter(x=prof.hour,y=prof.mean_kw,name=client))
    render_chart(style(fig,"Average power (kW)","Hour"))
    st.caption("Usage segments are quartiles calculated for the selected period, not machine-learned clusters.")
    st.dataframe(portfolio,hide_index=True,width="stretch",height=300)


PAGES={"Community Overview":overview,"Community Grid":community_grid,"DR Events":events,"Household Analytics":households}
