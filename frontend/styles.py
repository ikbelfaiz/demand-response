"""Central visual tokens for the light dashboard and Plotly charts."""

TEXT = {
    "primary": "#0F172A",
    "secondary": "#334155",
    "muted": "#64748B",
    "on_dark": "#F8FAFC",
}

SURFACES = {
    "app": "#F6F8FC",
    "card": "#FFFFFF",
    "subtle": "#F8FAFC",
    "sidebar": "#0F172A",
}

BORDERS = {"standard": "#E2E8F0", "strong": "#CBD5E1", "grid": "#E2E8F0"}

COLORS = {
    "household": "#2563EB", "zone": "#334155", "pv": "#059669",
    "event": "#D97706", "peak": "#DC2626", "production": "#7C3AED",
}

PLOTLY_THEME = {
    "font_family": "Inter, sans-serif",
    "primary_text": TEXT["primary"],
    "secondary_text": TEXT["secondary"],
    "muted_text": TEXT["muted"],
    "paper": SURFACES["card"],
    "plot": SURFACES["card"],
    "grid": BORDERS["grid"],
    "axis_line": BORDERS["strong"],
    "hover_background": TEXT["primary"],
    "hover_text": TEXT["on_dark"],
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --text-primary:{TEXT['primary']}; --text-secondary:{TEXT['secondary']};
  --text-muted:{TEXT['muted']}; --surface-app:{SURFACES['app']};
  --surface-card:{SURFACES['card']}; --border-standard:{BORDERS['standard']};
}}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
.stApp {{ background:var(--surface-app); color:var(--text-primary); }}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label {{ color:var(--text-secondary); }}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{ color:var(--text-primary); }}
[data-testid="stCaptionContainer"] p {{ color:#475569 !important; }}
[data-testid="stWidgetLabel"] p {{ color:var(--text-secondary) !important; font-weight:600; }}
[data-testid="stSidebar"] {{ background:{SURFACES['sidebar']}; }}
[data-testid="stSidebar"] * {{ color:#E2E8F0; }}
.hero {{ padding:1.9rem 2rem; border-radius:20px; color:white; background:linear-gradient(125deg,#0F172A 0%,#173C65 58%,#0F766E 120%); box-shadow:0 16px 36px rgba(15,23,42,.16); margin-bottom:1.2rem; }}
.hero h1 {{ font-size:2.15rem; margin:0 0 .35rem; letter-spacing:-.035em; color:#FFFFFF; }}
.hero h3 {{ margin:0 0 .75rem; color:#BAE6FD; font-size:1.08rem; font-weight:500; }}
.hero p {{ max-width:950px; margin:0; color:#DCEAF4 !important; line-height:1.6; }}
.section-title {{ font-size:1.35rem; font-weight:700; color:var(--text-primary); margin:1.4rem 0 .2rem; }}
.section-kicker {{ color:#475569; font-size:.9rem; margin-bottom:.75rem; }}
.status-strip {{ background:#ECFDF5; border:1px solid #A7F3D0; color:#065F46; border-radius:12px; padding:.7rem .9rem; margin:.2rem 0 1rem; font-size:.86rem; }}
.notice {{ background:#FFF7ED; border:1px solid #FDBA74; border-left:6px solid #D97706; border-radius:14px; padding:1rem 1.15rem; color:#7C2D12; }}
[data-testid="stMetric"] {{ background:var(--surface-card); border:1px solid var(--border-standard); border-radius:14px; padding:1rem 1.05rem; box-shadow:0 5px 18px rgba(15,23,42,.045); }}
[data-testid="stMetricLabel"] p {{ color:#475569 !important; font-weight:600; }}
[data-testid="stMetricValue"] {{ color:var(--text-primary) !important; }}
[data-testid="stMetricDelta"] {{ color:var(--text-secondary) !important; }}
.feature-card {{ background:var(--surface-card); border:1px solid var(--border-standard); border-radius:14px; padding:.85rem 1rem; color:#334155; }}
.scope-household {{ border-left:5px solid #2563EB; }}
.scope-regional {{ border-left:5px solid #059669; }}
.pipeline {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin:.6rem 0 1.2rem; }}
.pipeline-step {{ background:var(--surface-card); border:1px solid #DDE5EF; border-radius:14px; padding:1rem; min-height:92px; color:#334155; }}
.pipeline-step b {{ color:#0F766E; display:block; margin-bottom:.3rem; }}
.model-status {{ background:#F8FAFC; border:1px solid #CBD5E1; border-left:6px solid #64748B; border-radius:14px; padding:1.1rem; color:#334155; }}
.model-status b {{ color:var(--text-primary); }}
.page-label {{ color:#0F766E; font-weight:700; letter-spacing:.08em; font-size:.75rem; text-transform:uppercase; }}
[data-testid="stDataFrame"] {{ background:#FFFFFF; border:1px solid var(--border-standard); border-radius:12px; overflow:hidden; color-scheme:light; }}
[data-testid="stDataFrame"] * {{ color:var(--text-secondary); }}
button[data-baseweb="tab"] {{ color:var(--text-secondary) !important; font-weight:600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--text-primary) !important; }}
.footer {{ text-align:center; color:#64748B; padding:2rem 0 1rem; font-size:.82rem; }}
@media(max-width:900px){{.pipeline{{grid-template-columns:1fr 1fr}}}}
@media(max-width:700px){{.hero{{padding:1.35rem}}.hero h1{{font-size:1.7rem}}.pipeline{{grid-template-columns:1fr}}}}
</style>
"""
