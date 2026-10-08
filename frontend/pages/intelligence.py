import streamlit as st

from backend.intelligence.model_registry import ModelRegistry
from backend.services.historical_event_service import historical_event_labels
from frontend import charts
from frontend.components import page_header, render_chart, section


def render(full_data, filtered_data) -> None:
    page_header("Research roadmap", "Demand Response Intelligence", "Architecture for future model-driven grid-condition detection and operational decision support.")
    st.markdown('<div class="model-status"><b>Detection status</b><br>Demand Response detection model not yet implemented.<br><small>No live prediction or activation decision is being generated.</small></div>',unsafe_allow_html=True)
    section("Future Detection Pipeline","Detection, prediction and activation are deliberately separate responsibilities.")
    steps=[("1 • Data collection","Household, regional, production and context channels."),("2 • Feature extraction","Validated time, demand, production and weather features."),
           ("3 • Grid-condition analysis","Forecast peaks or detect unusual conditions; neither is an activation."),("4 • Detection model","Generate versioned candidate conditions with timestamps."),
           ("5 • Decision layer","Apply a separately governed policy to recommend DR activation."),("6 • Evaluation","Compare predictions with held-out labels and assess proposed response." )]
    st.markdown('<div class="pipeline">'+''.join(f'<div class="pipeline-step"><b>{a}</b>{b}</div>' for a,b in steps)+'</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.info("**Anomaly detection** identifies statistically unusual observations.")
    b.warning("**Peak prediction** forecasts high regional demand conditions.")
    c.success("**DR activation** is a separate operational decision with policy constraints.")
    section("Future AI Integration")
    registry=ModelRegistry()
    capabilities=[("Peak-demand forecasting","Interface ready; no model registered"),("Grid stress indicator","Interface ready; definition/model pending"),
                  ("DR event recommendation","Decision interface ready; policy pending"),("Household flexibility","Model and ground truth pending"),("DR optimization","Requires validated flexibility and objectives")]
    st.dataframe([{"Capability":name,"Current status":status} for name,status in capabilities],hide_index=True,width="stretch")
    st.caption(f"Registered runtime models: {len(registry.available())}. The dashboard runs independently of trained models.")
    section("Historical Dataset Labels","Optional retrospective exploration. These labels came from the CSV and are not model predictions, live alerts, or activation recommendations.")
    show=st.checkbox("Show historical DR labels for the selected period",value=False)
    if show:
        history=historical_event_labels(filtered_data)
        if history.empty:
            st.info("No historical dataset event labels occur in the selected period.")
        else:
            render_chart(charts.historical_labels(history))
            st.dataframe(history,hide_index=True,width="stretch")
    with st.expander("Full-dataset label inventory"):
        inventory=historical_event_labels(full_data)
        st.write(f"The source dataset contains {len(inventory)} labeled event windows. They are retained for future supervised evaluation only.")
