"""Selected-household demonstration pages."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data import load_events, load_household_info, load_participation
from backend.services import household_service
from backend.services.forecast_service import forecast_next_day, model_status
from backend.services.participation_service import for_household, response_counts
from frontend.charts import bars, donut, forecast_chart, heatmap, lines, style
from frontend.components import page_header, render_chart
from frontend.styles import COLORS


def energy(frame: pd.DataFrame, client_id: str, community_grid: pd.DataFrame) -> None:
    page_header("Consumer / demo", "My energy", f"Operational 30-minute measurements for {client_id}; not an authenticated account.")
    s=household_service.summary(frame); daily=household_service.daily_energy(frame); monthly=household_service.monthly_energy(frame)
    cols=st.columns(4)
    for col,(label,value) in zip(cols,[("Selected-period energy",f"{s['energy_kwh']:.1f} kWh"),("Average power",f"{s['average_kw']:.2f} kW"),("Peak power",f"{s['peak_kw']:.2f} kW"),("Load factor",f"{s['load_factor']:.1%}")]): col.metric(label,value)
    render_chart(lines(frame,[("aggregate_power_kw",client_id,COLORS["household"])],"Household power (kW)"))
    left,right=st.columns(2)
    with left: render_chart(bars(daily,"date","energy_kwh","Daily energy","Energy (kWh)"))
    with right: render_chart(bars(monthly,"month","energy_kwh","Monthly energy","Energy (kWh)",COLORS["pv"]))
    _forecast_section(client_id)
    left,right=st.columns(2)
    with left:
        prof=household_service.hourly_profile(frame); fig=go.Figure([go.Scatter(x=prof.hour,y=prof.mean_kw,name="Mean",line=dict(color=COLORS["household"])),go.Scatter(x=prof.hour,y=prof.p95_kw,name="P95",line=dict(color=COLORS["event"],dash="dash"))]); render_chart(style(fig,"Power (kW)","Hour"))
    with right: render_chart(heatmap(household_service.usage_heatmap(frame)))
    community_per_home=community_grid.households_consumption_kw/50
    st.metric("Community per-household mean",f"{community_per_home.mean():.2f} kW",help="Neighborhood feeder demand divided by 50; includes the modeled 3% feeder loss, so it is contextual rather than a meter-to-meter comparator.")


def _forecast_section(client_id: str) -> None:
    st.subheader("Tomorrow's Energy Forecast")
    st.caption("Saved Energy-TTM model: seven prior days (336 half-hours) predict the next 48 half-hours. Synthetic historical data, not a live smart-meter forecast.")
    available, status = model_status()
    if not available:
        st.warning(status)
        return
    left, right = st.columns(2)
    with left:
        mode = st.radio("Forecast mode", ["Historical held-out evaluation", "Next day after dataset"],
                        horizontal=True, key=f"forecast_mode_{client_id}")
    with right:
        if mode.startswith("Historical"):
            forecast_date = st.date_input(
                "Held-out forecast date", value=pd.Timestamp("2025-12-31").date(),
                min_value=pd.Timestamp("2025-11-01").date(), max_value=pd.Timestamp("2025-12-31").date(),
                key=f"forecast_date_{client_id}")
        else:
            forecast_date = pd.Timestamp("2026-01-01").date()
            st.text_input("Forecast date", value=str(forecast_date), disabled=True, key=f"next_date_{client_id}")
    if not st.button("Generate forecast", type="primary", key=f"generate_forecast_{client_id}"):
        st.info(f"Model ready: {status}. Choose a mode and generate a forecast.")
        return
    try:
        result = forecast_next_day(client_id, forecast_date)
    except Exception as exc:
        st.error(f"Forecast could not be generated: {exc}")
        return
    series, kpis = result["series"], result["kpis"]
    peak_time = series.iloc[int(kpis["peak_slot"])].timestamp
    cols = st.columns(4)
    metrics = [
        ("Predicted daily energy", f"{kpis['daily_energy_kwh']:.2f} kWh"),
        ("Predicted peak power", f"{kpis['peak_power_kw']:.2f} kW"),
        ("Predicted peak time", f"{peak_time:%H:%M}"),
        ("Predicted average power", f"{kpis['average_power_kw']:.2f} kW"),
    ]
    for col, (label, value) in zip(cols, metrics): col.metric(label, value)
    render_chart(forecast_chart(series))
    st.caption(
        f"Model version: {result['model_version']}. Forecasts are estimates; individual appliance "
        "events and their exact timing may not be predictable from aggregate consumption alone."
    )
    if result["actual_comparison"] == "held_out" and result["metrics"]:
        error = result["metrics"]
        actual_energy = float(series.actual_energy_kwh.sum())
        cols = st.columns(4)
        metrics = [
            ("MAE", f"{error['mae_kw']:.3f} kW"), ("RMSE", f"{error['rmse_kw']:.3f} kW"),
            ("WAPE", f"{error['wape_pct']:.1f}%"),
            ("Peak timing error", f"{error['peak_timing_error_minutes']:.0f} min"),
        ]
        for col, (label, value) in zip(cols, metrics): col.metric(label, value)
        cols = st.columns(3)
        cols[0].metric("Forecast daily energy", f"{kpis['daily_energy_kwh']:.2f} kWh")
        cols[1].metric("Actual daily energy", f"{actual_energy:.2f} kWh")
        cols[2].metric("Daily energy error", f"{error['daily_energy_error_kwh']:+.2f} kWh")
        st.caption("WAPE is total absolute forecast error divided by total actual consumption; it is undefined when actual daily consumption is zero.")
    elif result["actual_comparison"] == "unavailable":
        st.info("No actual measurements exist for this future date; only the forecast is shown.")
    else:
        st.warning("This date is in the model-development period and is not reported as held-out validation.")


def household(_frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    page_header("Consumer / profile", "My household", "Dataset attributes and optional shiftable-appliance demo inputs.")
    info=load_household_info().set_index("client_id").loc[client_id]
    cols=st.columns(3)
    values=[("Occupants",info.n_occupants),("Air conditioning","Yes" if info.has_ac else "No"),("Second AC","Yes" if info.has_second_ac else "No"),("Electric water heater","Yes" if info.has_electric_water_heater else "No"),("Washing machine","Yes" if info.has_washing_machine else "No"),("Daytime occupancy","Yes" if info.occupied_daytime else "No"),("Measured submeters","Yes" if info.has_submeter else "No")]
    for i,(label,value) in enumerate(values): cols[i%3].metric(label,value)
    st.info("The form below is demo-only session state and never changes the research dataset.")
    with st.form(f"onboarding_{client_id}"):
        appliances=st.multiselect("Appliances you are willing to shift",["Air conditioning","Electric water heater","Washing machine"],key=f"shift_{client_id}")
        willingness=st.slider("Participation willingness",0,100,60,key=f"willing_{client_id}")
        if st.form_submit_button("Save demo profile"): st.session_state.setdefault("demo_profiles",{})[client_id]={"appliances":appliances,"willingness":willingness}; st.success("Saved in this session only.")


def appliances(frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    page_header("Consumer / metering", "My appliances", "Measured submeter channels are shown only for panel households.")
    info=load_household_info().set_index("client_id").loc[client_id]
    if not info.has_submeter:
        st.warning("This household has aggregate metering only. Appliance estimates will be available after NILM training; none are fabricated here.")
        render_chart(lines(frame,[("aggregate_power_kw","Aggregate power",COLORS["household"])],"Power (kW)")); return
    series=[("aggregate_power_kw","Aggregate",COLORS["household"]),("ac_power_kw","AC submeter","#EF4444"),("water_heater_power_kw","Water heater submeter","#06B6D4"),("washing_machine_power_kw","Washing machine submeter","#8B5CF6")]
    render_chart(lines(frame,series,"Power (kW)"))
    rows=[]
    for c,label,_ in series:
        rows.append({"channel":label,"energy_kwh":frame[f"{c.removesuffix('_kw')}_energy_kwh"].sum(min_count=1)})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.caption("Channels are available for the ten panel households. Future NILM estimates for other households are not yet implemented.")


def participation(_frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    page_header("Consumer / history", "My DR participation", "Historical synthetic invitations and responses for the selected demo household.")
    history=for_household(load_participation(),load_events(),client_id); counts=response_counts(history)
    a,b,c,d=st.columns(4); a.metric("Invitations",len(history)); b.metric("Accepted",counts["accept"]); c.metric("Declined",counts["decline"]); d.metric("No response",counts["no_response"])
    render_chart(donut(["Accept","Decline","No response"],[counts["accept"],counts["decline"],counts["no_response"]],"Recorded responses"))
    st.dataframe(history[["event_id","start_ts","end_ts","response","reduction_target_pct"]],hide_index=True,width="stretch",height=350)
    st.subheader("Demo notification")
    st.caption("These controls only store a mock response in this browser session. Nothing is sent to STEG and the CSV is never modified.")
    choice=st.radio("Response",["Later","Accept","Decline"],horizontal=True,key=f"demo_response_{client_id}")
    if st.button("Save demo response",key=f"save_response_{client_id}"): st.session_state.setdefault("demo_responses",{})[client_id]=choice; st.success("Demo response saved in session state.")


PAGES={"My Energy":energy,"My Household":household,"My Appliances":appliances,"My DR Participation":participation}
