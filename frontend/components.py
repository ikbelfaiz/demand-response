import streamlit as st
import pandas as pd


def inject_styles(css: str) -> None:
    st.markdown(css, unsafe_allow_html=True)


def render_chart(figure) -> None:
    """Render figures without Streamlit overriding the centralized Plotly theme."""
    st.plotly_chart(figure, width="stretch", theme=None, config={
        "displayModeBar": False,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    })


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-kicker">{caption}</div>', unsafe_allow_html=True)


def unavailable(title: str, explanation: str) -> None:
    st.markdown(f'<div class="feature-card"><b>{title}</b><br>{explanation}</div>', unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="page-label">{kicker}</div><div class="section-title" style="margin-top:.25rem">{title}</div><div class="section-kicker">{description}</div>', unsafe_allow_html=True)


def period_selector(min_date, max_date) -> tuple[object, object, str]:
    mode = st.selectbox("Analysis period", ["Day", "Week", "Month", "Custom range"])
    max_ts, min_ts = pd.Timestamp(max_date), pd.Timestamp(min_date)
    if mode == "Day":
        chosen = st.date_input("Date", value=max_ts.date(), min_value=min_ts.date(), max_value=max_ts.date())
        return chosen, chosen, mode
    if mode == "Week":
        end = st.date_input("Week ending", value=max_ts.date(), min_value=min_ts.date(), max_value=max_ts.date())
        start = max(pd.Timestamp(end) - pd.Timedelta(days=6), min_ts).date()
        return start, end, mode
    if mode == "Month":
        months = pd.period_range(min_ts, max_ts, freq="M")
        chosen = st.selectbox("Month", list(months), index=len(months)-1, format_func=lambda p: p.strftime("%B %Y"))
        return max(chosen.start_time, min_ts).date(), min(chosen.end_time, max_ts).date(), mode
    selected = st.date_input("Custom dates", value=(max(max_ts-pd.Timedelta(days=6),min_ts).date(),max_ts.date()), min_value=min_ts.date(),max_value=max_ts.date())
    if isinstance(selected, tuple) and len(selected)==2:
        return selected[0], selected[1], mode
    return selected, selected, mode
