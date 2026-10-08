"""Streamlit entry point for the intelligent energy monitoring platform."""
import pandas as pd
import streamlit as st

from backend.config import get_dataset_config
from backend.data import load_dataset, prepare_dataset, validate_dataset
from backend.services.analytics_service import filter_dates
from frontend.components import inject_styles, period_selector
from frontend.pages import household, intelligence, overview, quality, regional
from frontend.styles import CSS

st.set_page_config(page_title="Intelligent Energy & DR Platform", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner="Loading 2025 energy measurements…")
def cached_dataset(path: str, size: int, modified: int):
    config=get_dataset_config()
    raw=load_dataset(config)
    return prepare_dataset(raw,config),validate_dataset(raw,config)


def application_data():
    config=get_dataset_config()
    if not config.path.is_file():
        raise FileNotFoundError(f"Dataset not found: {config.path}")
    stat=config.path.stat()
    return cached_dataset(str(config.path),stat.st_size,stat.st_mtime_ns)


def main() -> None:
    inject_styles(CSS)
    st.markdown("""
    <div class="hero"><h1>⚡ Intelligent Energy & Demand Response Platform</h1>
    <h3>Household monitoring • Regional grid analytics • Future AI decision support</h3>
    <p>A measurement-driven view of the 2025 C001 household and TUN regional energy dataset.
    Historical labels remain separate from future model predictions and operational decisions.</p></div>
    """,unsafe_allow_html=True)
    try:
        data,report=application_data()
    except (OSError,ValueError,pd.errors.ParserError) as error:
        st.error(f"The 2025 dataset could not be loaded: {error}")
        st.stop()
    pages={"Energy Overview":overview.render,"Household Analysis":household.render,"Regional Grid":regional.render,
           "DR Intelligence":intelligence.render,"Data Quality":quality.render}
    with st.sidebar:
        st.markdown("## ⚡ Platform navigation")
        page=st.radio("Workspace",list(pages),label_visibility="collapsed")
        st.markdown("### Time range")
        start,end,mode=period_selector(data.timestamp.min().date(),data.timestamp.max().date())
        st.markdown("---")
        st.caption("Source: dr_dataset_2025_1min_v2.csv")
        st.caption("Native resolution: 1 minute • timezone unspecified")
    filtered=filter_dates(data,start,end)
    st.markdown(f'<div class="status-strip"><b>{page}</b> • {pd.Timestamp(start):%d %b %Y} to {pd.Timestamp(end):%d %b %Y} • {len(filtered):,} one-minute rows</div>',unsafe_allow_html=True)
    if page=="DR Intelligence":
        intelligence.render(data,filtered)
    elif page=="Data Quality":
        quality.render(data,filtered,report)
    else:
        pages[page](filtered)
    st.markdown('<div class="footer">Intelligent Energy Monitoring & Demand Response Research Platform</div>',unsafe_allow_html=True)


if __name__=="__main__":
    main()
