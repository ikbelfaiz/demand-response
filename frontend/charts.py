"""Accessible Plotly builders shared by every dashboard."""
import plotly.graph_objects as go
import pandas as pd
from .styles import COLORS, PLOTLY_THEME


def style(fig: go.Figure, y_title: str = "", x_title: str = "Time", height: int = 390) -> go.Figure:
    axis = dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(color="#475569"),
                title_font=dict(color="#334155"), automargin=True)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", hovermode="x unified",
        font=dict(family=PLOTLY_THEME["font_family"], color="#334155"),
        title_font=dict(color="#0F172A"), legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(color="#334155")),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#F8FAFC"),
        xaxis={**axis, "title":x_title}, yaxis={**axis, "title":y_title})
    return fig


def lines(frame: pd.DataFrame, series: list[tuple[str, str, str]], y_title: str, height: int = 410) -> go.Figure:
    fig = go.Figure()
    for column, label, color in series:
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.timestamp, y=frame[column], name=label, connectgaps=False,
                line=dict(color=color,width=2),hovertemplate="%{x|%d %b %H:%M}<br>%{y:.3f} kW<extra></extra>"))
    return style(fig, y_title, height=height)


def bars(frame: pd.DataFrame, x: str, y: str, label: str, y_title: str, color: str = COLORS["household"]) -> go.Figure:
    return style(go.Figure(go.Bar(x=frame[x], y=frame[y], name=label, marker_color=color)), y_title, x.replace("_", " ").title())


def heatmap(matrix: pd.DataFrame, unit: str = "kW") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=[str(v) for v in matrix.index], colorscale="Blues",
        colorbar=dict(title=unit, tickfont=dict(color="#475569")), hovertemplate="%{y}<br>%{x}:00<br>%{z:.2f} "+unit+"<extra></extra>"))
    return style(fig, "Date", "Hour", 430)


def donut(labels, values, title: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.62, marker_colors=["#0F766E", "#D97706", "#64748B", "#2563EB"]))
    fig.update_layout(title=title)
    return style(fig, "", "", 330)


def forecast_chart(frame: pd.DataFrame) -> go.Figure:
    """Render actual and forecast as separate, high-contrast traces."""
    fig = go.Figure()
    if "actual_power_kw" in frame:
        fig.add_trace(go.Scatter(
            x=frame.timestamp, y=frame.actual_power_kw, name="Actual power",
            mode="lines+markers", connectgaps=False,
            line=dict(color="#2563EB", width=2), marker=dict(size=4),
            hovertemplate="%{x|%d %b %H:%M}<br>Actual: %{y:.3f} kW<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=frame.timestamp, y=frame.predicted_power_kw, name="Energy-TTM forecast",
        mode="lines+markers", connectgaps=False,
        line=dict(color="#7C3AED", width=2, dash="dash"), marker=dict(size=4),
        hovertemplate="%{x|%d %b %H:%M}<br>Forecast: %{y:.3f} kW<extra></extra>",
    ))
    return style(fig, "Household power (kW)", "Forecast time", 430)


def dr_replay_chart(frame: pd.DataFrame, risk_windows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    traces = [
        ("actual_demand_kw","Actual observed demand","#0F172A","solid"),
        ("actual_supply_kw","Actual modeled supply","#64748B","solid"),
        ("predicted_demand_kw","Predicted community demand","#2563EB","dash"),
        ("predicted_supply_kw","Predicted total available supply","#059669","dash"),
    ]
    for column,label,color,dash in traces:
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.timestamp,y=frame[column],name=label,connectgaps=False,
                line=dict(color=color,width=2,dash=dash),hovertemplate=f"%{{x|%d %b %H:%M}}<br>{label}: %{{y:.2f}} kW<extra></extra>"))
    for number,row in enumerate(risk_windows.itertuples(),1):
        fig.add_vrect(x0=row.start_ts,x1=row.end_ts,fillcolor="#F97316",opacity=.14,line_width=0,
                      annotation_text="Forecast risk" if number==1 else None,annotation_position="top left")
    return style(fig,"Power (kW)","Target-day time",500)
