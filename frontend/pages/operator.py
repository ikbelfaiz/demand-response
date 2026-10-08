"""Operator perspective pages for the synthetic neighborhood."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data import load_events, load_household, load_household_info, load_participation, operational_household
from backend.services import community_service, event_service, grid_service, household_service
from backend.services.dr_replay_service import held_out_dates, replay_model_status, run_replay
from backend.services.segmentation_service import consumption_quantiles
from frontend.charts import bars, donut, dr_replay_chart, heatmap, lines, style
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


def dr_detection(_grid: pd.DataFrame) -> None:
    page_header("Operator / AI research", "DR Detection & Forecasting",
        "Leakage-safe D-1 14:00 historical replay. Recommendations are simulated research outputs, never operational activations.")
    available,status=replay_model_status()
    if not available:
        st.warning(status); return
    dates=held_out_dates()
    left,right=st.columns([2,1])
    with left:
        chosen=st.date_input("Historical replay target day",value=dates[-1].date(),
            min_value=dates[0].date(),max_value=dates[-1].date(),key="dr_replay_date")
    with right:
        st.metric("Saved model",status)
    st.caption("The selector is restricted to the July–August historical replay period. Forecast inputs stop at 14:00 on the prior day.")
    if not st.button("Run Forecast / Replay",type="primary",key="run_dr_replay"):
        st.info("Select a historical date and run the saved-model replay. No model is trained on page load."); return
    try: result=run_replay(chosen)
    except Exception as exc: st.error(f"Historical replay failed: {exc}"); return
    comparison=result["comparison"]; risks=result["candidate_peak_windows"]; recommendations=result["recommended_dr_events"]
    st.success(result["status"])
    st.caption(f"Issued {result['prediction_issue_timestamp']:%d %b %Y %H:%M} • target {result['target_day']:%d %b %Y} • {result['model_version']}")
    deficit=(-comparison.predicted_margin_kw).clip(lower=0)
    peak_slot=int(comparison.predicted_demand_kw.to_numpy().argmax())
    kpis=[("Predicted peak demand",f"{comparison.predicted_demand_kw.max():.1f} kW"),
          ("Predicted peak time",f"{comparison.timestamp.iloc[peak_slot]:%H:%M}"),
          ("Minimum forecast margin",f"{comparison.predicted_margin_kw.min():.1f} kW"),
          ("Maximum expected shortage",f"{deficit.max():.1f} kW"),
          ("Risk intervals",str(int(deficit.gt(0).sum()))),("Expected shortage energy",f"{deficit.sum()*.5:.1f} kWh")]
    cols=st.columns(3)
    for i,(label,value) in enumerate(kpis): cols[i%3].metric(label,value)
    render_chart(dr_replay_chart(comparison,risks))
    st.subheader("Recommended DR events")
    st.caption("Simulated decision generated by the model during historical replay.")
    if recommendations.empty: st.info("The fixed research policy recommends no event for this day.")
    else: st.dataframe(recommendations[["start_ts","end_ts","max_deficit_kw","shortage_energy_kwh",
        "duration_minutes","recommended_reduction_kw","decision_status","reason"]],hide_index=True,width="stretch")
    st.subheader("Historical context")
    st.caption("Observed demand and historical events are shown for context and are not treated as shortage truth.")
    st.caption("Historical actual traces use the completed canonical series; original-source coverage remains documented in offline evaluation outputs.")
    if not result["historical_operator_events"].empty:
        st.dataframe(result["historical_operator_events"],hide_index=True,width="stretch")


PAGES={"Community Overview":overview,"Community Grid":community_grid,"DR Events":events,
       "Household Analytics":households,"DR Detection & Forecasting":dr_detection}
