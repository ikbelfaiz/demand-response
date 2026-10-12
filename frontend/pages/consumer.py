"""Selected-household demonstration pages."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data import load_events, load_household_info, load_participation
from backend.services import household_service
from backend.services import segmentation_service as seg
from backend.services.forecast_service import consumer_forecast_dates, forecast_next_day, model_status

from backend.services.forecast_service import forecast_next_day, model_status
from backend.services.nilm_service import (APPLIANCES as NILM_APPLIANCES, DISPLAY_NAMES,
                                           export_csv as nilm_export_csv,
                                           model_status as nilm_model_status,
                                           predict_appliances)
 
from backend.services.participation_service import for_household, response_counts
from frontend.charts import bars, donut, forecast_chart, heatmap, lines, style
from frontend.components import page_header, render_chart
from frontend.pages import dr_orchestrator
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
    st.subheader("Daily Energy Forecast")
    st.caption("Select an eligible 2025 date. The saved Energy-TTM model uses its seven prior days (336 half-hours) to predict that day's 48 half-hours.")
    available, status = model_status()
    if not available:
        st.warning(status)
        return
    dates = consumer_forecast_dates(client_id)
    if dates.empty:
        st.warning("No forecast date has the complete seven-day context required by this artifact.")
        return
    forecast_date = st.date_input(
        "Historical forecast date", value=dates.max().date(),
        min_value=dates.min().date(), max_value=dates.max().date(),
        key=f"forecast_date_{client_id}",
        help=f"Available from {dates.min().date()} after seven complete historical days.",
    )
    if not st.button("Generate forecast", type="primary", key=f"generate_forecast_{client_id}"):
        st.info(f"Model ready: {status}. Choose a target date and generate a forecast.")
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
    st.caption(result["forecast_status_label"])
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
        actual_energy = float(series.actual_energy_kwh.sum())
        st.metric("Actual daily energy", f"{actual_energy:.2f} kWh")
        st.caption("Independent forecast-error metrics are hidden for retrospective model-development dates.")


def household(_frame: pd.DataFrame, client_id: str, _grid: pd.DataFrame) -> None:
    page_header("Consumer / profile", "My household", "Dataset attributes and optional shiftable-appliance demo inputs.")
    info=load_household_info().set_index("client_id").loc[client_id]
    cols=st.columns(3)
    values=[("Occupants",info.n_occupants),("Air conditioning","Yes" if info.has_ac else "No"),("Second AC","Yes" if info.has_second_ac else "No"),("Electric water heater","Yes" if info.has_electric_water_heater else "No"),("Washing machine","Yes" if info.has_washing_machine else "No"),("Daytime occupancy","Yes" if info.occupied_daytime else "No"),("Measured submeters","Yes" if info.has_submeter else "No")]
    for i,(label,value) in enumerate(values): cols[i%3].metric(label,value)
    _segment_card(client_id)
    st.info("The form below is demo-only session state and never changes the research dataset.")
    with st.form(f"onboarding_{client_id}"):
        appliances=st.multiselect("Appliances you are willing to shift",["Air conditioning","Electric water heater","Washing machine"],key=f"shift_{client_id}")
        willingness=st.slider("Participation willingness",0,100,60,key=f"willing_{client_id}")
        if st.form_submit_button("Save demo profile"): st.session_state.setdefault("demo_profiles",{})[client_id]={"appliances":appliances,"willingness":willingness}; st.success("Saved in this session only.")


def _segment_card(client_id: str) -> None:
    st.subheader("My consumption profile")
    if not seg.profiling_available():
        st.info(f"Consumption profile not available yet. Operator must run: {seg.TRAIN_COMMAND}"); return
    hh = seg.household_segments(); me = hh.loc[hh.client_id.eq(client_id)].iloc[0]
    a, b, c = st.columns(3)
    a.metric("My segment", me.segment)
    b.metric("My usual evening use (18–22h)", f"{me.usual_evening_kwh:.1f} kWh/day",
             f"{me.vs_similar_households_pct:+.0f}% vs similar homes", delta_color="inverse")
    c.metric("My DR acceptance rate", f"{me.acceptance_rate:.0%}")
    st.success(me.comparison_message)
    load = seg.segment_load_profiles()
    left, right = st.columns(2)
    season = left.radio("Season", ["summer", "autumn", "winter", "spring"], horizontal=True, key=f"my_season_{client_id}")
    day_type = right.radio("Day type", ["weekday", "weekend"], horizontal=True, key=f"my_day_{client_id}")
    d_ = load[(load.segment == me.segment) & (load.season == season) & (load.day_type == day_type)].sort_values("slot")
    fig = go.Figure(go.Scatter(x=d_.slot / 2, y=d_.mean_kw_per_household, name=f"Typical curve: {me.segment}",
                               line=dict(color=COLORS["household"], width=2.5)))
    render_chart(style(fig, "Average power per household (kW)", "Hour of day"))

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
    if seg.profiling_available():
        costs=seg.event_costs(client_id=client_id)[["event_id","window_kwh","peak_price_millimes","event_cost_dt","surcharge_dt"]]
        rew=seg.rewards(client_id=client_id)[["event_id","reward_dt"]]
        history=history.merge(costs,on="event_id",how="left").merge(rew,on="event_id",how="left").fillna({"reward_dt":0.0})
        x,y,z=st.columns(3); x.metric("Total cost of events",f"{history.event_cost_dt.sum():.2f} DT"); y.metric("Of which peak surcharge",f"{history.surcharge_dt.sum():.2f} DT"); z.metric("Rewards earned",f"{history.reward_dt.sum():.2f} DT")
        cols=["event_id","start_ts","end_ts","response","window_kwh","peak_price_millimes","event_cost_dt","surcharge_dt","reward_dt"]
        num=["window_kwh","event_cost_dt","surcharge_dt","reward_dt"]; view=history[cols].copy(); view[num]=view[num].round(3)
        st.dataframe(view,hide_index=True,width="stretch",height=350)
        st.caption("Cost of an event = kWh consumed in the window × peak price (band price +50%). Rewards are your share of your segment's measured savings.")
    else:
        st.dataframe(history[["event_id","start_ts","end_ts","response","reduction_target_pct"]],hide_index=True,width="stretch",height=350)
    dr_orchestrator.participation_section(client_id)


PAGES={"My Energy":energy,"My Household":household,"My Appliances":appliances,"My DR Assistant":dr_orchestrator.assistant,
       "My DR Participation":participation}
