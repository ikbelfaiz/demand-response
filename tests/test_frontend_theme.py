import pandas as pd
from frontend.charts import heatmap, lines


def test_chart_text_is_dark_on_light():
    frame=pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=2,freq="30min"),"x":[1,2]})
    fig=lines(frame,[("x","Demand","#2563EB")],"Power (kW)")
    assert fig.layout.paper_bgcolor=="#FFFFFF"; assert fig.layout.font.color=="#334155"
    assert fig.layout.xaxis.tickfont.color=="#475569"; assert fig.layout.yaxis.gridcolor=="#E2E8F0"


def test_heatmap_uses_accessible_colorbar():
    fig=heatmap(pd.DataFrame([[1.]],index=["2025-01-01"],columns=[0])); assert fig.data[0].colorbar.tickfont.color=="#475569"


def test_line_chart_has_no_gap_overlays_or_auxiliary_traces():
    frame=pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=3,freq="30min"),"power_kw":[1.0,2.0,0.0]})
    fig=lines(frame,[("power_kw","Power","#2563EB")],"kW")
    assert len(fig.data)==1 and len(fig.layout.shapes)==0
    assert not pd.isna(fig.data[0].y).any() and fig.data[0].y[2]==0.0
