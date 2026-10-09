"""Selected-household demonstration pages."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data import load_events, load_household_info, load_participation
from backend.services import household_service
from backend.services.forecast_service import forecast_next_day, model_status
from backend.services.nilm_service import (APPLIANCES as NILM_APPLIANCES, DISPLAY_NAMES,
                                           export_csv as nilm_export_csv,
                                           model_status as nilm_model_status,
                                           predict_appliances)
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
    if result["actual_comparison"] == "held_out" and result["metrics"]:
        error = result["metrics"]
        cols = st.columns(4)
        metrics = [
            ("MAE", f"{error['mae_kw']:.3f} kW"), ("RMSE", f"{error['rmse_kw']:.3f} kW"),
            ("WAPE", f"{error['wape_pct']:.1f}%"),
            ("Daily energy error", f"{error['daily_energy_error_kwh']:+.2f} kWh"),
        ]
        for col, (label, value) in zip(cols, metrics): col.metric(label, value)
        st.caption("WAPE is total absolute forecast error divided by total actual consumption; it is undefined when actual daily consumption is zero.")
    elif result["actual_comparison"] == "unavailable":
        st.info("No actual measurements exist for this future date; only the forecast is shown.")
    else:
        st.warning("This date is in the model-development period and is not reported as held-out validation.")


def household(_frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    page_header("Consumer / profile", "My household", "Dataset attributes and isolated demonstration preferences.")
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
    page_header("Consumer / NILM research", "Appliance estimates",
                "Causal TCN estimates from native one-minute aggregate power; historical 2025 data only.")
    available, status = nilm_model_status()
    if not available:
        st.error(f"NILM model unavailable: {status}")
        return
    metadata = status
    st.caption(f"Model {metadata['model_identifier']} • TCN • {metadata['context_length']}-minute context • "
               f"device: {metadata['execution_device']} • timestamps: Africa/Tunis civil time")
    st.info("Research estimates trained on synthetic data. AC means primary AC only; washing-machine estimates are experimental. "
            "Activity probabilities are not calibrated confidence, and pooled metrics do not establish performance on new or unlabeled households.")

    dataset_min, dataset_max = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:59")
    default_start = max(pd.Timestamp(frame.timestamp.min()) if len(frame) else dataset_min, dataset_min)
    default_end = min(default_start + pd.Timedelta(days=1), pd.Timestamp("2026-01-01"))
    c1, c2, c3, c4 = st.columns(4)
    with c1: start_date = st.date_input("Start date", default_start.date(), dataset_min.date(), dataset_max.date(), key=f"nilm_sd_{client_id}")
    with c2: start_time = st.time_input("Start time", default_start.time(), step=60, key=f"nilm_st_{client_id}")
    with c3: end_date = st.date_input("End date", default_end.date(), dataset_min.date(), pd.Timestamp("2026-01-01").date(), key=f"nilm_ed_{client_id}")
    with c4: end_time = st.time_input("End time (exclusive)", default_end.time(), step=60, key=f"nilm_et_{client_id}")
    start = pd.Timestamp.combine(start_date, start_time)
    end = pd.Timestamp.combine(end_date, end_time)
    key = f"nilm_result_{client_id}"
    if st.button("Analyze consumption", type="primary", key=f"nilm_analyze_{client_id}"):
        try:
            with st.spinner("Running causal TCN inference on one-minute readings…"):
                st.session_state[key] = predict_appliances(client_id, start, end)
        except Exception as exc:
            st.session_state.pop(key, None)
            st.error(f"Analysis could not be completed: {exc}")
    result = st.session_state.get(key)
    if result is None:
        st.caption("Choose an interval and select Analyze consumption. Requests are limited to seven days by default.")
        return

    summary, minute, hourly = result.summary, result.minute, result.hourly
    st.subheader("Coverage and quality")
    cols = st.columns(4)
    cols[0].metric("Requested minutes", f"{summary['expected_minutes']:,}")
    cols[1].metric("Available readings", f"{summary['available_readings']:,}")
    cols[2].metric("Valid predictions", f"{summary['valid_predicted_minutes']:,}")
    cols[3].metric("Prediction coverage", f"{summary['coverage_fraction']:.1%}")
    if summary["excluded_minutes_by_reason"]:
        st.warning("Excluded minutes: " + ", ".join(f"{reason}: {count}" for reason, count in summary["excluded_minutes_by_reason"].items()))
    st.caption(f"Inference latency for this cached result: {result.latency_seconds:.3f} s. Missing predictions remain null and energy is never extrapolated.")

    st.subheader("Appliance summary")
    colors = {"ac": "#EF4444", "water_heater": "#06B6D4", "washing_machine": "#8B5CF6"}
    for col, appliance in zip(st.columns(3), NILM_APPLIANCES):
        item = summary["appliances"][appliance]
        with col:
            st.markdown(f"**{item['display_name']}**")
            energy = item["observed_estimated_energy_kwh"]
            st.metric("Observed estimated energy", "Unavailable" if energy is None else f"{energy:.3f} kWh")
            st.metric("Coverage", f"{item['coverage_fraction']:.1%}")
            latest = item["latest_valid_timestamp"]
            st.metric("Latest estimated power", "Unavailable" if latest is None else f"{item['latest_valid_power_w']:.1f} W")
            if latest is not None:
                st.caption(f"At {latest:%Y-%m-%d %H:%M} Africa/Tunis • activity probability {item['latest_activity_probability']:.1%}")

    show_reference = st.checkbox("Show measured reference overlays when available", value=False,
                                 key=f"nilm_refs_{client_id}")
    plot = minute.set_index("timestamp")
    averaged = len(plot) > 1440
    if averaged:
        numeric = ["aggregate_power_w", *[f"predicted_{a}_power_w" for a in NILM_APPLIANCES],
                   *[f"reference_{a}_power_w" for a in NILM_APPLIANCES]]
        quality = plot.quality_status.resample("15min").apply(lambda values: values.eq("ok").all())
        plot = plot[numeric].resample("15min").mean()
        plot.loc[~quality, [f"predicted_{a}_power_w" for a in NILM_APPLIANCES]] = float("nan")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot.index, y=plot.aggregate_power_w, name="Aggregate", connectgaps=False,
                             line=dict(color=COLORS["household"], width=2)))
    for appliance in NILM_APPLIANCES:
        fig.add_trace(go.Scatter(x=plot.index, y=plot[f"predicted_{appliance}_power_w"],
                                 name=f"Estimated {DISPLAY_NAMES[appliance]}", connectgaps=False,
                                 line=dict(color=colors[appliance], width=1.7)))
        reference = f"reference_{appliance}_power_w"
        if show_reference and reference in plot and plot[reference].notna().any():
            fig.add_trace(go.Scatter(x=plot.index, y=plot[reference], name=f"Reference {DISPLAY_NAMES[appliance]}",
                                     connectgaps=False, line=dict(color=colors[appliance], dash="dot")))
    render_chart(style(fig, "Power (W)", "Africa/Tunis time", 470))
    if averaged: st.caption("Power display is averaged to 15-minute bins; inference and energy calculations remain one-minute resolution.")

    activity = go.Figure()
    for appliance in NILM_APPLIANCES:
        activity.add_trace(go.Scatter(x=minute.timestamp, y=minute[f"{appliance}_activity_probability"],
                                      name=DISPLAY_NAMES[appliance], connectgaps=False,
                                      line=dict(color=colors[appliance])))
        ref = f"reference_{appliance}_activity"
        if show_reference and minute[ref].notna().any():
            activity.add_trace(go.Scatter(x=minute.timestamp, y=minute[ref], name=f"Reference {DISPLAY_NAMES[appliance]}",
                                          connectgaps=False, line=dict(color=colors[appliance], dash="dot")))
    activity.update_yaxes(range=[0, 1])
    render_chart(style(activity, "Activity probability", "Africa/Tunis time", 340))

    energy_fig = go.Figure()
    for appliance in NILM_APPLIANCES:
        energy_fig.add_trace(go.Bar(x=hourly.hour, y=hourly[f"{appliance}_energy_kwh"],
                                    name=DISPLAY_NAMES[appliance], marker_color=colors[appliance],
                                    customdata=hourly[["valid_minutes", "requested_minutes", "coverage_fraction"]],
                                    hovertemplate="%{x}<br>%{y:.3f} kWh<br>Coverage %{customdata[2]:.1%} (%{customdata[0]}/%{customdata[1]} requested min)<extra></extra>"))
    energy_fig.update_layout(barmode="group")
    render_chart(style(energy_fig, "Observed estimated energy (kWh)", "Hour (Africa/Tunis)", 410))
    st.caption("Hourly bars sum valid minute-average power only. Coverage is relative to the requested part of each hour; partial hours are not extrapolated.")

    st.subheader("Minute detail and export")
    st.dataframe(minute, hide_index=True, width="stretch", height=380)
    left, right = st.columns(2)
    left.download_button("Download minute predictions CSV", nilm_export_csv(result, "minute"),
                         f"nilm_{client_id}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}_minute.csv", "text/csv")
    right.download_button("Download hourly summary CSV", nilm_export_csv(result, "hourly"),
                          f"nilm_{client_id}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}_hourly.csv", "text/csv")
    metrics = metadata.get("verified_pooled_test_metrics", {})
    if metrics:
        st.caption("Saved pooled test artifacts — " + "; ".join(
            f"{DISPLAY_NAMES[name]}: MAE {values['power_mae_w']:.1f} W, activity F1 {values['activity_f1']:.3f}"
            for name, values in metrics.items()) + ". F1 is not overall accuracy.")


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


def preferences(_frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    page_header("Consumer / settings", "Preferences", "Local demonstration settings designed for future account-backed storage.")
    saved=st.session_state.setdefault("preferences",{}).get(client_id,{})
    with st.form(f"prefs_{client_id}"):
        language=st.selectbox("Preferred language",["English","French","Arabic"],index=["English","French","Arabic"].index(saved.get("language","English")))
        quiet=st.slider("Quiet hours",0,23,(saved.get("quiet_start",22),saved.get("quiet_end",7)))
        notifications=st.multiselect("Notifications",["DR invitations","High-use alerts","Weekly summary"],default=saved.get("notifications",["DR invitations"]))
        willingness=st.select_slider("General participation willingness",["Low","Conditional","High"],value=saved.get("willingness","Conditional"))
        shifts=st.multiselect("Shiftable appliances",["Air conditioning","Water heater","Washing machine"],default=saved.get("shifts",[]))
        if st.form_submit_button("Save preferences"):
            st.session_state["preferences"][client_id]={"language":language,"quiet_start":quiet[0],"quiet_end":quiet[1],"notifications":notifications,"willingness":willingness,"shifts":shifts}; st.success("Saved locally for this session.")


PAGES={"My Energy":energy,"My Household":household,"My Appliances":appliances,"My DR Participation":participation,"Preferences":preferences}
