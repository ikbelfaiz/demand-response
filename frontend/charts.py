import plotly.graph_objects as go
import pandas as pd

from .styles import COLORS, PLOTLY_THEME


def style(fig: go.Figure, y_title: str, height: int = 390, x_title: str = "Time") -> go.Figure:
    theme = PLOTLY_THEME
    axis = dict(
        showgrid=True, gridcolor=theme["grid"], gridwidth=1,
        showline=True, linecolor=theme["axis_line"], linewidth=1,
        zeroline=True, zerolinecolor=theme["axis_line"],
        tickfont=dict(family=theme["font_family"], size=12, color=theme["secondary_text"]),
        automargin=True,
    )
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=20,r=20,t=38,b=20),
        paper_bgcolor=theme["paper"], plot_bgcolor=theme["plot"], hovermode="x unified",
        font=dict(family=theme["font_family"], size=12, color=theme["secondary_text"]),
        title_font=dict(family=theme["font_family"], size=16, color=theme["primary_text"]),
        legend=dict(orientation="h",y=1.04,x=1,xanchor="right",bgcolor="rgba(255,255,255,.92)",
                    bordercolor=theme["grid"],borderwidth=1,
                    font=dict(family=theme["font_family"],size=12,color=theme["primary_text"])),
        hoverlabel=dict(bgcolor=theme["hover_background"],bordercolor=theme["axis_line"],
                        font=dict(family=theme["font_family"],size=13,color=theme["hover_text"])),
        annotationdefaults=dict(font=dict(family=theme["font_family"],size=12,color=theme["primary_text"]),
                                bgcolor="rgba(255,255,255,.94)",bordercolor=theme["grid"],borderwidth=1),
        xaxis={**axis,"title":dict(text=x_title,font=dict(family=theme["font_family"],size=13,color=theme["primary_text"]))},
        yaxis={**axis,"title":dict(text=y_title,font=dict(family=theme["font_family"],size=13,color=theme["primary_text"]))},
    )
    return fig


def household_load(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Scatter(x=data.timestamp,y=data.household_power_kw,name="Household demand",line=dict(color=COLORS["household"],width=2)))
    return style(fig,"Household power (kW)",430)


def regional_demand(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Scatter(x=data.timestamp,y=data.zone_demand_mw,name="Regional demand",line=dict(color=COLORS["zone"],width=2)))
    return style(fig,"Regional power (MW)",400)


def production_demand(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure()
    for col,label,color in (("zone_demand_mw","Regional demand",COLORS["zone"]),("system_production_mw","STEG production",COLORS["production"]),("zone_pv_production_mw","Regional PV",COLORS["pv"])):
        if col in data:
            fig.add_trace(go.Scatter(x=data.timestamp,y=data[col],name=label,line=dict(color=color,width=2)))
    return style(fig,"Regional power (MW)",420)


def pv_production(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Scatter(x=data.timestamp,y=data.zone_pv_production_mw,name="Regional PV",fill="tozeroy",
                             fillcolor="rgba(16,185,129,.12)",line=dict(color=COLORS["pv"],width=2)))
    return style(fig,"PV production (MW)",350)


def daily_energy(data: pd.DataFrame) -> go.Figure:
    custom=data.observed_coverage if "observed_coverage" in data else None
    fig=go.Figure(go.Bar(x=data.date,y=data.energy_kwh,name="Integrated household energy",marker_color=COLORS["household"],customdata=custom,
                         hovertemplate="%{x}<br>%{y:.2f} kWh<br>Source coverage %{customdata:.1%}<extra></extra>"))
    return style(fig,"Integrated energy (kWh)",360,"Date")


def hourly_pattern(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data.hour,y=data.average_power_kw,name="Mean",line=dict(color=COLORS["household"],width=3)))
    fig.add_trace(go.Scatter(x=data.hour,y=data.p95_power_kw,name="95th percentile",line=dict(color=COLORS["event"],width=2,dash="dash")))
    return style(fig,"Household power (kW)",360,"Hour of day")


def appliance_stack(data: pd.DataFrame, labels: dict[str,str]) -> go.Figure:
    palette=["#EF4444","#06B6D4","#8B5CF6","#F59E0B","#64748B"]
    fig=go.Figure()
    for (column,label),color in zip(labels.items(),palette):
        if column in data:
            fig.add_trace(go.Scatter(x=data.timestamp,y=data[column],name=label,stackgroup="appliances",line=dict(color=color,width=.6)))
    return style(fig,"Measured appliance power (kW)",390)


def appliance_energy(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Bar(x=data.appliance,y=data.energy_kwh,marker_color=["#EF4444","#06B6D4","#8B5CF6"][:len(data)],
                         customdata=data.coverage,hovertemplate="%{x}<br>%{y:.2f} kWh<br>Coverage %{customdata:.1%}<extra></extra>"))
    return style(fig,"Observed energy (kWh)",350,"Appliance")


def load_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Heatmap(z=matrix.values,x=matrix.columns,y=[str(v) for v in matrix.index],colorscale="Blues",
                             colorbar=dict(title=dict(text="kW",font=dict(color=PLOTLY_THEME["primary_text"])),
                                           tickfont=dict(color=PLOTLY_THEME["secondary_text"]),
                                           outlinecolor=PLOTLY_THEME["axis_line"]),
                             hovertemplate="Date %{y}<br>Hour %{x}:00<br>%{z:.2f} kW<extra></extra>"))
    return style(fig,"Date",420,"Hour of day")


def demand_distribution(values: pd.Series) -> go.Figure:
    fig=go.Figure(go.Histogram(x=values,nbinsx=50,marker_color=COLORS["household"],name="Demand readings"))
    return style(fig,"Count",350,"Household power (kW)")


def regional_daily(data: pd.DataFrame) -> go.Figure:
    return production_demand(data)


def historical_labels(data: pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Bar(x=data.start,y=data.duration_minutes,name="Dataset event label",marker_color=COLORS["event"],
                         customdata=data[["peak_zone_mw","peak_household_kw"]],
                         hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y} min<br>Zone peak %{customdata[0]:.1f} MW<br>Household peak %{customdata[1]:.2f} kW<extra></extra>"))
    return style(fig,"Label duration (minutes)",350,"Historical label start")
