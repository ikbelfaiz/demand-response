"""Demand Response v3 Streamlit application."""
import pandas as pd
import streamlit as st

from backend.config import canonical_data_signature
from backend.data import (assert_v3_files, filter_time, load_grid, load_household,
                          load_household_info, operational_grid, operational_household)
from frontend.components import inject_styles, period_selector
from frontend.pages import consumer, operator
from frontend.styles import CSS

st.set_page_config(page_title="Demand Response v3", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner="Loading v3 community measurements…")
def community_data(signature):
    raw=load_grid(); return raw, operational_grid(raw)


@st.cache_data(show_spinner="Loading household smart-meter measurements…")
def household_data(client_id: str, start: pd.Timestamp, end: pd.Timestamp, signature):
    return operational_household(load_household(client_id,start,end))


def main() -> None:
    inject_styles(CSS); assert_v3_files(); signature=canonical_data_signature(); raw_grid, op_grid=community_data(signature)
    st.markdown('<div class="hero"><h1>⚡ Intelligent Demand Response & Community Energy</h1><h3>Operator intelligence • Household insight • Research-ready infrastructure</h3><p>A synthetic 50-household Tunis neighborhood case study. Historical events and modeled supply assumptions are clearly separated from future AI capabilities.</p></div>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## Platform workspace")
        perspective=st.radio("Perspective",["Operator Dashboard","Consumer Dashboard"],key="perspective")
        pages=operator.PAGES if perspective.startswith("Operator") else consumer.PAGES
        page=st.radio("Page",list(pages),key=f"page_{perspective}")
        st.markdown("### Analysis period")
        start_date,end_date,_=period_selector(raw_grid.timestamp.min().date(),raw_grid.timestamp.max().date())
        client_id=None
        if perspective.startswith("Consumer"):
            ids=load_household_info().client_id.tolist(); client_id=st.selectbox("Demo household",ids,key="client_id")
        st.markdown("---"); st.caption("Synthetic academic demonstration — not authentication or an operational STEG system."); st.caption("Canonical source: complete 1-minute synthetic v3 → 30-minute operational view • timestamps: naive local civil time")
    start=pd.Timestamp(start_date); end=pd.Timestamp(end_date)+pd.DateOffset(days=1)
    grid=filter_time(op_grid,start,end)
    st.markdown(f'<div class="status-strip"><b>{perspective} · {page}</b> • {start:%d %b %Y} to {end-pd.DateOffset(days=1):%d %b %Y} • operational 30-minute layer</div>',unsafe_allow_html=True)
    if perspective.startswith("Operator"):
        if page=="Household Analytics": pages[page](grid,start,end)
        else: pages[page](grid)
    else:
        household=household_data(client_id,start,end,signature); pages[page](household,client_id,grid)
    st.markdown('<div class="footer">Demand Response v3 • Synthetic neighborhood research platform • Energy-TTM forecasts use saved artifacts only</div>',unsafe_allow_html=True)


if __name__=="__main__": main()
