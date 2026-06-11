"""
Hyderabad Metro Last-Mile Intelligence Platform
================================================
Premium urban mobility intelligence dashboard.
Reads pre-computed outputs only — does NOT regenerate analytics.

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HYD Metro Accessibility Platform",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────

ACCENT   = "#00D4FF"
CRITICAL = "#FF4B4B"
HIGH     = "#FF8C00"
MEDIUM   = "#FFD700"
LOW      = "#00C48C"
MUTED    = "#8B9AB2"
PURPLE   = "#6366F1"

# Metro line colours (Hyderabad HMRL)
LINE_COLORS = {
    "Red":   "#E8003D",
    "Blue":  "#1B4FD8",
    "Green": "#00843D",
}

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — Premium dark intelligence platform
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
[data-testid="stAppViewContainer"] {
    background: #0A0E1A;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(0,212,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(99,102,241,0.04) 0%, transparent 50%);
}
[data-testid="stSidebar"] {
    background: #080C17;
    border-right: 1px solid rgba(30,42,58,0.8);
    box-shadow: 4px 0 24px rgba(0,0,0,0.4);
}
[data-testid="stHeader"]  { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }

/* ── Typography ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1 { color: #FFFFFF !important; font-size: 1.55rem !important; font-weight: 800 !important;
     letter-spacing: -0.03em; margin-bottom: 0 !important; }
h2 { color: #E2E8F0 !important; font-size: 1.05rem !important; font-weight: 600 !important; letter-spacing: -0.01em; }
h3 { color: #64748B !important; font-size: 0.72rem !important; font-weight: 600 !important;
     text-transform: uppercase; letter-spacing: 0.1em; margin: 0; }

/* ── Metric cards — premium ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0F1626 0%, #111827 100%);
    border: 1px solid #1E2D40;
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,255,0.25);
    box-shadow: 0 0 20px rgba(0,212,255,0.06);
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; font-weight: 600 !important; }

/* ── Containers ── */
[data-testid="stContainer"] { border-radius: 12px; }
[data-testid="stVerticalBlock"] > [data-testid="stContainer"][data-border="true"] {
    background: linear-gradient(135deg, #0D1420 0%, #111827 100%);
    border: 1px solid #1A2535;
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0D1420;
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid #1E2D40;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 6px;
    padding: 6px 16px !important;
    letter-spacing: 0.02em;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1E293B !important;
    color: #00D4FF !important;
    border-bottom-color: transparent !important;
}

/* ── Sidebar elements ── */
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { display: none; }

/* Style radio buttons as nav items */
[data-testid="stRadio"] > div { gap: 2px !important; }
[data-testid="stRadio"] label {
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    transition: background 0.15s !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    cursor: pointer;
}
[data-testid="stRadio"] label:hover { background: rgba(30,41,59,0.6) !important; }
[data-testid="stRadio"] label[data-checked="true"] {
    background: rgba(0,212,255,0.08) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
}
/* Hide default radio circles */
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0D1420;
    border: 1px solid #1E2D40;
    border-radius: 8px;
    color: #E2E8F0;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1A2535;
}
[data-testid="stDataFrame"] thead th {
    background: #0A1020 !important;
    color: #475569 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #1E2D40 !important;
}
[data-testid="stDataFrame"] tbody tr { transition: background 0.1s; }
[data-testid="stDataFrame"] tbody tr:hover td { background: #131E30 !important; }
[data-testid="stDataFrame"] tbody td {
    color: #CBD5E1 !important;
    font-size: 0.82rem !important;
    border-color: #0F1A28 !important;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label { color: #94A3B8 !important; font-size: 0.8rem !important; }

/* ── Dividers ── */
hr { border-color: #1A2535 !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent;
    border: 1px solid #1E2D40;
    color: #64748B;
    font-size: 0.78rem;
    border-radius: 8px;
    padding: 6px 16px;
    transition: all 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #00D4FF;
    color: #00D4FF;
    background: rgba(0,212,255,0.05);
}

/* ── Badge pills ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(255,75,75,0.12);  color: #FF4B4B; border: 1px solid rgba(255,75,75,0.3); }
.badge-high     { background: rgba(255,140,0,0.12);  color: #FF8C00; border: 1px solid rgba(255,140,0,0.3); }
.badge-medium   { background: rgba(255,215,0,0.10);  color: #FFD700; border: 1px solid rgba(255,215,0,0.3); }
.badge-low      { background: rgba(0,196,140,0.10);  color: #00C48C; border: 1px solid rgba(0,196,140,0.3); }
.badge-monitor  { background: rgba(139,154,178,0.10); color: #8B9AB2; border: 1px solid rgba(139,154,178,0.3); }

/* ── Insight callout cards ── */
.insight-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.05) 0%, rgba(99,102,241,0.03) 100%);
    border: 1px solid rgba(0,212,255,0.15);
    border-left: 3px solid #00D4FF;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin: 6px 0;
    font-size: 0.84rem;
    color: #CBD5E1;
    line-height: 1.5;
}
.insight-card strong { color: #00D4FF; }

.warning-card {
    border-left-color: #FF8C00;
    border-color: rgba(255,140,0,0.15);
    background: linear-gradient(135deg, rgba(255,140,0,0.04) 0%, transparent 100%);
}
.warning-card strong { color: #FF8C00; }

.critical-card {
    border-left-color: #FF4B4B;
    border-color: rgba(255,75,75,0.15);
    background: linear-gradient(135deg, rgba(255,75,75,0.04) 0%, transparent 100%);
}
.critical-card strong { color: #FF4B4B; }

/* ── Section headers ── */
.section-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #CBD5E1;
    letter-spacing: 0.01em;
    margin-bottom: 2px;
}
.section-subtitle {
    font-size: 0.72rem;
    color: #334155;
    margin-bottom: 12px;
}

/* ── Page title block ── */
.page-title-block {
    border-bottom: 1px solid #1A2535;
    padding-bottom: 14px;
    margin-bottom: 20px;
}
.page-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #00D4FF;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.83rem;
    color: #475569;
    margin-top: 4px;
}

/* ── Station deploy card ── */
.deploy-card {
    background: linear-gradient(135deg, #0D1420 0%, #111827 100%);
    border: 1px solid #1A2535;
    border-left: 3px solid #FF4B4B;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.deploy-card:hover {
    border-color: rgba(255,75,75,0.5);
    box-shadow: 0 2px 16px rgba(255,75,75,0.08);
}
.deploy-name { color: #F1F5F9; font-weight: 700; font-size: 0.88rem; }
.deploy-action { color: #64748B; font-size: 0.78rem; margin-top: 2px; }
.deploy-score { color: #475569; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; }

/* ── Ranked station card ── */
.rank-card {
    background: #0D1420;
    border: 1px solid #1A2535;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.15s;
}
.rank-card:hover { border-color: #1E3A5F; }
.rank-num {
    font-size: 0.72rem;
    font-weight: 800;
    color: #1E3A5F;
    font-family: 'JetBrains Mono', monospace;
    width: 28px;
    text-align: right;
    flex-shrink: 0;
}
.rank-bar-wrap { flex: 1; }
.rank-name { color: #CBD5E1; font-size: 0.82rem; font-weight: 600; }
.rank-score { color: #475569; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }
.rank-bar { height: 3px; border-radius: 2px; margin-top: 5px; background: rgba(0,212,255,0.15); position: relative; overflow: hidden; }
.rank-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #00D4FF, #6366F1); }

/* ── Progress bar ── */
.prog-bar { height: 4px; background: #1E2D40; border-radius: 2px; overflow: hidden; margin-top: 6px; }
.prog-fill { height: 100%; border-radius: 2px; }

/* ── Scenario comparison card ── */
.scenario-card {
    background: linear-gradient(135deg, #0D1420, #0F1828);
    border: 1px solid #1A2535;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s, transform 0.15s;
}
.scenario-card:hover { border-color: rgba(0,212,255,0.2); transform: translateY(-1px); }
.scenario-type {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #00D4FF; margin-bottom: 6px;
}
.scenario-name { font-size: 0.9rem; font-weight: 700; color: #F1F5F9; }
.scenario-metrics { display: flex; gap: 20px; margin-top: 10px; }
.scenario-metric-val { font-size: 1.15rem; font-weight: 800; font-family: 'Inter', sans-serif; }
.scenario-metric-lbl { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Network impact prose card ── */
.network-impact {
    background: #0A1020;
    border: 1px solid #1A2535;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
}
.network-impact-label {
    color: #334155;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.network-impact-text {
    color: #94A3B8;
    font-size: 0.85rem;
    line-height: 1.7;
}

/* ── Map legend pill ── */
.map-legend {
    background: rgba(8,12,23,0.92);
    border: 1px solid #1A2535;
    border-radius: 10px;
    padding: 12px 16px;
    display: inline-block;
}
.legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.78rem; color: #94A3B8; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* ── Mono values ── */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* ── Hide Streamlit chrome ── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING — cached, robust
# ─────────────────────────────────────────────────────────────

OUTPUTS = Path("outputs")

@st.cache_data(show_spinner=False)
def _load(filename: str) -> pd.DataFrame | None:
    path = OUTPUTS / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None


def load_all():
    return {
        "exec_summary":    _load("executive_summary_metrics.csv"),
        "priority_scores": _load("station_priority_scores.csv"),
        "mismatch":        _load("demand_service_mismatch.csv"),
        "insights_top5":   _load("conversion_insights_top5.csv"),
        "mclp_coverage":   _load("mclp_coverage_by_k.csv"),
        "mclp_selected":   _load("mclp_selected_stations.csv"),
        "mclp_candidates": _load("mclp_candidate_scores.csv"),
        "sim_impacts":     _load("simulation_station_impacts.csv"),
        "sim_ranking":     _load("simulation_intervention_ranking.csv"),
        "sim_network":     _load("simulation_network_summary.csv"),
        "sim_scenarios":   _load("simulation_scenarios.csv"),
        "station_coords":  _load("station_coordinates.csv"),
    }


# ─────────────────────────────────────────────────────────────
# DESIGN HELPERS
# ─────────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#64748B", family="Inter, sans-serif", size=11),
    xaxis=dict(gridcolor="#0F1828", linecolor="#1E293B", zerolinecolor="#1E293B",
               tickfont=dict(size=10, color="#475569")),
    yaxis=dict(gridcolor="#0F1828", linecolor="#1E293B", zerolinecolor="#1E293B",
               tickfont=dict(size=10, color="#475569")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                font=dict(size=10, color="#64748B")),
    margin=dict(l=4, r=4, t=28, b=4),
)


# ── Plotly layout helpers — prevent duplicate-keyword crashes ──────────────
def base_layout_without(*keys) -> dict:
    """Return PLOTLY_THEME with the given top-level keys removed."""
    return {k: v for k, v in PLOTLY_THEME.items() if k not in keys}


def axis_layout(axis_name: str, **overrides) -> dict:
    """Merge PLOTLY_THEME[axis_name] with caller overrides (no duplication)."""
    return {**PLOTLY_THEME.get(axis_name, {}), **overrides}


def plotly_layout(**overrides) -> dict:
    """
    Return a layout dict that is PLOTLY_THEME merged with *overrides*.
    Keys present in overrides are removed from the base first so that
    update_layout(**plotly_layout(...)) never receives duplicate kwargs.
    """
    base = PLOTLY_THEME.copy()
    for k in overrides:
        base.pop(k, None)
    return {**base, **overrides}
# ──────────────────────────────────────────────────────────────────────────

COLOR_MAP_BAND = {
    "Critical": CRITICAL,
    "High":     HIGH,
    "Medium":   MEDIUM,
    "Low":      LOW,
    "Monitor":  MUTED,
}


def _safe_val(df, col, default=0):
    if df is None or col not in df.columns:
        return default
    v = df[col].iloc[0]
    return v if pd.notna(v) else default


def _fmt_num(n, decimals=0):
    try:
        if decimals == 0:
            return f"{int(round(float(n))):,}"
        return f"{float(n):,.{decimals}f}"
    except Exception:
        return str(n)


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<div class='section-title'>{title}</div>"
        + (f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""),
        unsafe_allow_html=True,
    )


def page_title(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f"<div class='page-title-block'>"
        f"<div class='page-eyebrow'>{eyebrow}</div>"
        f"<h1>{title}</h1>"
        f"<div class='page-subtitle'>{subtitle}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def insight_card(text: str, variant: str = "default"):
    cls = {"warning": "warning-card", "critical": "critical-card"}.get(variant, "")
    st.markdown(
        f"<div class='insight-card {cls}'>{text}</div>",
        unsafe_allow_html=True,
    )


def notice(msg: str):
    st.markdown(
        f"<div style='background:#0D1420;border:1px solid #1A2535;border-left:3px solid #1E2D40;"
        f"border-radius:0 8px 8px 0;padding:10px 14px;font-size:0.8rem;color:#334155;"
        f"margin:4px 0'>⚠ {msg}</div>",
        unsafe_allow_html=True,
    )


def badge_html(band: str) -> str:
    cls = f"badge-{band.lower()}" if band.lower() in ["critical","high","medium","low","monitor"] else "badge-monitor"
    return f"<span class='badge {cls}'>{band}</span>"


def rank_card_html(rank: int, name: str, score: float, max_score: float, color: str = ACCENT) -> str:
    pct = max(4, int((score / max_score) * 100)) if max_score else 4
    return (
        f"<div class='rank-card'>"
        f"<div class='rank-num'>#{rank:02d}</div>"
        f"<div class='rank-bar-wrap'>"
        f"<div class='rank-name'>{name}</div>"
        f"<div class='rank-bar'><div class='rank-bar-fill' style='width:{pct}%'></div></div>"
        f"</div>"
        f"<div class='rank-score'>{score:.1f}</div>"
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Premium navigation
# ─────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("🔷", "Executive Overview",       "System pulse & KPIs"),
    ("🗺",  "Transit Intelligence Map", "Spatial connectivity layer"),
    ("🔬", "Station Intelligence",      "Per-station diagnostics"),
    ("⚙️", "Optimization & Coverage",  "MCLP facility placement"),
    ("🧪", "Scenario Simulation",       "Intervention impact models"),
    ("🚀", "Intervention Planning",    "Indicative deployment guidance"),
]

def render_sidebar():
    with st.sidebar:
        # Logo block
        st.markdown("""
        <div style='padding: 1.4rem 0.5rem 1rem'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
                <div style='width:32px;height:32px;border-radius:8px;
                     background:linear-gradient(135deg,#00D4FF22,#6366F122);
                     border:1px solid rgba(0,212,255,0.3);
                     display:flex;align-items:center;justify-content:center;
                     font-size:16px'>🚇</div>
                <div>
                    <div style='color:#FFFFFF;font-weight:800;font-size:1rem;
                         letter-spacing:-0.02em;line-height:1'>METRO IQ</div>
                    <div style='color:#334155;font-size:0.65rem;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.1em'>Hyderabad · v2.0</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,#1E2A3A,transparent);margin-bottom:12px'></div>", unsafe_allow_html=True)

        # Navigation label
        st.markdown("<div style='color:#1E3A5F;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;padding:0 4px;margin-bottom:6px'>Navigation</div>", unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [item[1] for item in NAV_ITEMS],
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,#1E2A3A,transparent);margin:14px 0'></div>", unsafe_allow_html=True)

        # System status
        st.markdown("""
        <div style='padding:10px 12px;background:#080C17;border:1px solid #1A2535;border-radius:10px'>
            <div style='color:#1E3A5F;font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>System Status</div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>
                <div style='width:6px;height:6px;border-radius:50%;background:#00C48C;box-shadow:0 0 6px #00C48C'></div>
                <span style='color:#475569;font-size:0.72rem'>Analytics Engine</span>
                <span style='color:#00C48C;font-size:0.68rem;margin-left:auto'>LIVE</span>
            </div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>
                <div style='width:6px;height:6px;border-radius:50%;background:#00C48C;box-shadow:0 0 6px #00C48C'></div>
                <span style='color:#475569;font-size:0.72rem'>LMCI Model</span>
                <span style='color:#00C48C;font-size:0.68rem;margin-left:auto'>READY</span>
            </div>
            <div style='display:flex;align-items:center;gap:8px'>
                <div style='width:6px;height:6px;border-radius:50%;background:#00C48C;box-shadow:0 0 6px #00C48C'></div>
                <span style='color:#475569;font-size:0.72rem'>MCLP Solver</span>
                <span style='color:#00C48C;font-size:0.68rem;margin-left:auto'>READY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:14px;color:#1E2D40;font-size:0.65rem;text-align:center;line-height:1.6'>
            Urban Mobility Accessibility & Decision Support<br>
            Hyderabad Metro · Last-Mile Platform
        </div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────

def page_executive_overview(data: dict):
    page_title(
        "01 / System Overview",
        "Executive Intelligence Briefing",
        "Network-wide accessibility diagnostics · Hyderabad Metro HMRL · Multimodal connectivity intelligence",
    )

    exec_df  = data["exec_summary"]
    priority = data["priority_scores"]
    insights = data["insights_top5"]

    # ── KPI ROW ──────────────────────────────────────────────
    if exec_df is not None and not exec_df.empty:
        with st.container(horizontal=True):
            st.metric(
                "Stations Analysed",
                _fmt_num(_safe_val(exec_df, "total_stations_scored")),
                "Full network", border=True,
            )
            n_crit = _safe_val(exec_df, "critical_priority_stations")
            st.metric("Highest-Priority Stations", _fmt_num(n_crit), "Recommended for evaluation", border=True)
            st.metric("Elevated-Priority Stations", _fmt_num(_safe_val(exec_df, "high_priority_stations")), "Near-term review window", border=True)
            n_des = _safe_val(exec_df, "persistent_transit_deserts")
            st.metric("Low-Access Zones", _fmt_num(n_des), "Candidate recovery areas", border=True)
            cov = _safe_val(exec_df, "mclp_coverage_pct_at_k5", np.nan)
            cov_str = f"{cov:.1f}%" if pd.notna(cov) else "N/A"
            st.metric("Modelled Coverage @k=5", cov_str, "Scenario-based estimate", border=True)
    else:
        notice("executive_summary_metrics.csv not found. Run the full pipeline first.")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── EXECUTIVE INSIGHT PANELS ─────────────────────────────
    if priority is not None and not priority.empty:
        n_crit_stations = int((priority.get("priority_band", pd.Series()) == "Critical").sum()) if "priority_band" in priority.columns else 0
        n_total = len(priority)
        n_deserts = int(priority.get("is_persistent_desert", pd.Series(False)).astype(str).str.lower().isin(["true","1","yes"]).sum()) if "is_persistent_desert" in priority.columns else 0

        icols = st.columns(3, gap="small")
        with icols[0]:
            insight_card(
                f"<strong>{n_crit_stations} stations</strong> score in the highest accessibility-risk band "
                f"and are recommended for priority feasibility assessment.",
                "critical",
            )
        with icols[1]:
            insight_card(
                f"<strong>{n_deserts} low-access zones</strong> identified — concentrated in "
                f"expansion corridors with limited feeder connectivity. Requires ground-level validation.",
                "warning",
            )
        with icols[2]:
            insight_card(
                "Integrated <strong>multimodal hubs</strong> show the highest relative accessibility uplift "
                "per intervention unit across all modelled scenarios.",
            )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── CHARTS ROW ───────────────────────────────────────────
    left, right = st.columns([1.65, 1], gap="medium")

    with left:
        with st.container(border=True):
            section_header("Intervention Priority Distribution", "Station count by planning tier — composite accessibility scoring")
            if priority is not None and "priority_band" in priority.columns:
                band_counts = (
                    priority["priority_band"]
                    .value_counts()
                    .reindex(["Critical", "High", "Medium", "Low", "Monitor"], fill_value=0)
                    .reset_index()
                )
                band_counts.columns = ["Band", "Stations"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=band_counts["Band"],
                    y=band_counts["Stations"],
                    marker=dict(
                        color=[COLOR_MAP_BAND.get(b, MUTED) for b in band_counts["Band"]],
                        opacity=0.9,
                        line=dict(width=0),
                    ),
                    text=band_counts["Stations"],
                    textposition="outside",
                    textfont=dict(size=13, color="#CBD5E1", family="Inter"),
                    width=0.55,
                ))
                fig.update_layout(
                    **{k: v for k, v in PLOTLY_THEME.items() if k not in ("xaxis", "yaxis")},
                    showlegend=False,
                    height=240,
                    yaxis=dict(**PLOTLY_THEME["yaxis"], title=None),
                    xaxis=dict(**PLOTLY_THEME["xaxis"], title=None),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("Priority data unavailable.")

    with right:
        with st.container(border=True):
            section_header("Demand–Service Mismatch", "Alignment classification across network")
            if priority is not None and "mismatch_class" in priority.columns:
                mc = priority["mismatch_class"].value_counts().reset_index()
                mc.columns = ["Class", "Count"]
                fig2 = go.Figure(go.Pie(
                    labels=mc["Class"],
                    values=mc["Count"],
                    hole=0.62,
                    marker=dict(
                        colors=[CRITICAL, HIGH, MEDIUM, LOW, MUTED],
                        line=dict(color="#0A0E1A", width=2),
                    ),
                    textinfo="percent",
                    textfont=dict(size=10, color="#94A3B8"),
                    hovertemplate="<b>%{label}</b><br>%{value} stations (%{percent})<extra></extra>",
                ))
                fig2.add_annotation(
                    text=f"<b>{mc['Count'].sum()}</b><br><span style='font-size:9px'>STATIONS</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color="#F1F5F9", size=14, family="Inter"),
                )
                fig2.update_layout(**plotly_layout(
                    showlegend=True, height=240,
                    legend=axis_layout("legend", orientation="v", x=1.02),
                ))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("Mismatch class data unavailable.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── TOP PRIORITY STATIONS — Ranked cards ─────────────────
    with st.container(border=True):
        section_header("Top 10 Priority Stations", "Composite accessibility score · Indicative planning tier")
        if priority is not None and "final_priority_score" in priority.columns:
            top10 = priority.nlargest(10, "final_priority_score").reset_index(drop=True)
            max_score = top10["final_priority_score"].max()
            cards_html = ""
            for i, row in top10.iterrows():
                name  = str(row.get("stop_name", "—"))
                score = float(row.get("final_priority_score", 0))
                band  = str(row.get("priority_band", "Monitor"))
                intv  = str(row.get("recommended_intervention", ""))
                pct   = max(4, int((score / max_score) * 100)) if max_score else 4
                bar_color = COLOR_MAP_BAND.get(band, MUTED)
                cards_html += (
                    f"<div class='rank-card'>"
                    f"<div class='rank-num'>#{i+1:02d}</div>"
                    f"<div class='rank-bar-wrap' style='flex:1'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                    f"<span class='rank-name'>{name}</span>"
                    f"{badge_html(band)}"
                    f"</div>"
                    f"<div style='color:#334155;font-size:0.72rem;margin-top:2px'>{intv}</div>"
                    f"<div class='rank-bar'><div class='rank-bar-fill' style='width:{pct}%;background:{bar_color}'></div></div>"
                    f"</div>"
                    f"<div class='rank-score'>{score:.1f}</div>"
                    f"</div>"
                )
            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            notice("station_priority_scores.csv not found.")

    # ── CONVERSION OPPORTUNITIES — compact ───────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        section_header("Top Intervention Candidates", "Highest accessibility uplift potential — indicative prioritisation")
        if insights is not None and not insights.empty:
            cols_show = [c for c in [
                "stop_name", "final_priority_score", "recommended_intervention",
                "deploy_action", "priority_band",
            ] if c in insights.columns]
            if cols_show:
                disp = insights[cols_show].copy()
                disp.rename(columns={
                    "stop_name": "Station", "final_priority_score": "Priority Score",
                    "recommended_intervention": "Indicative Action",
                    "deploy_action": "Planning Status", "priority_band": "Band",
                }, inplace=True)
                if "Score" in disp.columns:
                    disp["Score"] = disp["Score"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—")
                st.dataframe(disp, hide_index=True, use_container_width=True)
            else:
                st.dataframe(insights, hide_index=True, use_container_width=True)
        else:
            notice("conversion_insights_top5.csv not found.")


# ─────────────────────────────────────────────────────────────
# PAGE 2 — TRANSIT INTELLIGENCE MAP
# ─────────────────────────────────────────────────────────────

def page_transit_map(data: dict):
    page_title(
        "02 / Spatial Intelligence",
        "Transit Intelligence Map",
        "Spatial accessibility layer · Planning priority zones · Low-access area overlays · Demand signal mapping",
    )

    priority = data["priority_scores"]

    if priority is None or priority.empty:
        notice("station_priority_scores.csv not found. Run the pipeline first.")
        return

    for col in ["stop_lat", "stop_lon"]:
        if col in priority.columns:
            priority[col] = pd.to_numeric(priority[col], errors="coerce")
    map_df = priority.dropna(subset=["stop_lat", "stop_lon"]).copy()

    if map_df.empty:
        notice("No valid station coordinates found.")
        return

    # ── MAP LAYER CONTROLS ────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='height:1px;background:linear-gradient(90deg,transparent,#1E2A3A,transparent);margin:10px 0 12px'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#1E3A5F;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>Map Layers</div>",
            unsafe_allow_html=True,
        )
        show_all      = st.checkbox("All Stations",      value=True)
        show_critical = st.checkbox("Critical Priority", value=True)
        show_mclp     = st.checkbox("MCLP Optimised",   value=True)
        show_deserts  = st.checkbox("Transit Deserts",   value=True)
        show_heatmap  = st.checkbox("Priority Heatmap",  value=False)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        map_style = st.selectbox(
            "Map Style",
            ["carto-darkmatter", "dark", "satellite-streets"],
            index=0,
            label_visibility="collapsed",
        )

    # Build derived columns
    def _color(row):
        if "priority_band" in row.index:
            return COLOR_MAP_BAND.get(row["priority_band"], MUTED)
        return MUTED

    map_df["_color"] = map_df.apply(_color, axis=1)
    map_df["_size"]  = map_df.get("final_priority_score", pd.Series(50, index=map_df.index)).fillna(50)

    # Rich hover template
    def _hover(row):
        band  = row.get("priority_band", "—")
        score = row.get("final_priority_score", "—")
        mism  = row.get("mismatch_class", "—")
        desert = "Yes" if str(row.get("is_persistent_desert","")).lower() in ["true","1","yes"] else "No"
        intv  = row.get("recommended_intervention", "—")
        score_str = f"{float(score):.1f}" if score != "—" and pd.notna(score) else "—"
        return (
            f"<b style='color:#F1F5F9'>{row.get('stop_name','')}</b><br>"
            f"<span style='color:#64748B'>Priority Band: </span>{band}<br>"
            f"<span style='color:#64748B'>Score: </span>{score_str}<br>"
            f"<span style='color:#64748B'>Mismatch: </span>{mism}<br>"
            f"<span style='color:#64748B'>Transit Desert: </span>{desert}<br>"
            f"<span style='color:#64748B'>Action: </span>{intv}"
        )

    map_df["_hover"] = map_df.apply(_hover, axis=1)

    fig = go.Figure()

    # ── Layer: All stations — sized by priority ───────────────
    if show_all:
        fig.add_trace(go.Scattermapbox(
            lat=map_df["stop_lat"],
            lon=map_df["stop_lon"],
            mode="markers",
            marker=dict(
                size=map_df["_size"].clip(8, 20) / 2.2,
                color=map_df["_color"],
                opacity=0.75,
            ),
            hovertext=map_df["_hover"],
            hoverinfo="text",
            name="All Stations",
            showlegend=True,
        ))

    # ── Layer: Priority heatmap ───────────────────────────────
    if show_heatmap and "final_priority_score" in map_df.columns:
        fig.add_trace(go.Densitymapbox(
            lat=map_df["stop_lat"],
            lon=map_df["stop_lon"],
            z=map_df["final_priority_score"].fillna(0),
            radius=40,
            colorscale=[[0, "rgba(0,0,0,0)"], [0.4, "rgba(255,75,75,0.1)"], [1, "rgba(255,75,75,0.5)"]],
            showscale=False,
            name="Priority Heatmap",
        ))

    # ── Layer: Critical — glowing rings ──────────────────────
    if show_critical and "priority_band" in map_df.columns:
        crit = map_df[map_df["priority_band"] == "Critical"]
        if not crit.empty:
            # Outer glow ring
            fig.add_trace(go.Scattermapbox(
                lat=crit["stop_lat"], lon=crit["stop_lon"],
                mode="markers",
                marker=dict(size=22, color=CRITICAL, opacity=0.15),
                hoverinfo="skip",
                name="",
                showlegend=False,
            ))
            # Inner marker
            fig.add_trace(go.Scattermapbox(
                lat=crit["stop_lat"], lon=crit["stop_lon"],
                mode="markers+text",
                text=crit.get("stop_name", pd.Series()),
                textposition="top right",
                textfont=dict(size=9, color=CRITICAL, family="Inter"),
                marker=dict(size=13, color=CRITICAL, opacity=1.0),
                hovertext=crit["_hover"],
                hoverinfo="text",
                name="Critical Priority",
                showlegend=True,
            ))

    # ── Layer: MCLP — star markers ────────────────────────────
    if show_mclp and "mclp_selected" in map_df.columns:
        mclp = map_df[map_df["mclp_selected"].astype(str).str.lower().isin(["true","1","yes"])]
        if not mclp.empty:
            fig.add_trace(go.Scattermapbox(
                lat=mclp["stop_lat"], lon=mclp["stop_lon"],
                mode="markers",
                marker=dict(size=14, color=ACCENT, opacity=0.95, symbol="star"),
                hovertext=mclp["_hover"],
                hoverinfo="text",
                name="MCLP Optimised",
                showlegend=True,
            ))

    # ── Layer: Transit deserts ─────────────────────────────────
    if show_deserts and "is_persistent_desert" in map_df.columns:
        deserts = map_df[map_df["is_persistent_desert"].astype(str).str.lower().isin(["true","1","yes"])]
        if not deserts.empty:
            # Desert zone glow
            fig.add_trace(go.Scattermapbox(
                lat=deserts["stop_lat"], lon=deserts["stop_lon"],
                mode="markers",
                marker=dict(size=24, color="#FF006E", opacity=0.08),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(
                lat=deserts["stop_lat"], lon=deserts["stop_lon"],
                mode="markers",
                marker=dict(size=11, color="#FF006E", opacity=0.85, symbol="square"),
                hovertext=deserts["_hover"],
                hoverinfo="text",
                name="Transit Deserts",
                showlegend=True,
            ))

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            center=dict(lat=17.385, lon=78.486),
            zoom=10.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(8,12,23,0.88)",
            bordercolor="#1E2A3A",
            borderwidth=1,
            font=dict(color="#94A3B8", size=11, family="Inter"),
            x=0.01, y=0.98,
            xanchor="left",
        ),
        height=640,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    # ── MAP STAT STRIP ────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    mcols = st.columns(4, gap="small")

    n_crit = int((map_df.get("priority_band", pd.Series()) == "Critical").sum()) if "priority_band" in map_df.columns else "—"
    n_mclp = int(map_df.get("mclp_selected", pd.Series(False)).astype(str).str.lower().isin(["true","1","yes"]).sum())
    n_des  = int(map_df.get("is_persistent_desert", pd.Series(False)).astype(str).str.lower().isin(["true","1","yes"]).sum()) if "is_persistent_desert" in map_df.columns else "—"

    with mcols[0]: st.metric("Stations Mapped",    len(map_df), border=True)
    with mcols[1]: st.metric("Highest-Risk Nodes", n_crit, "Evaluation recommended", border=True)
    with mcols[2]: st.metric("MCLP Candidates",    n_mclp, "Modelled placement sites", border=True)
    with mcols[3]: st.metric("Low-Access Zones",   n_des,  "Candidate recovery areas", border=True)

    # ── SPATIAL INSIGHTS ──────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            section_header("Planning Tier Distribution", "Station count by accessibility priority band")
            if "priority_band" in map_df.columns:
                band_counts = map_df["priority_band"].value_counts().reset_index()
                band_counts.columns = ["Band", "Count"]
                fig3 = go.Figure(go.Bar(
                    x=band_counts["Count"],
                    y=band_counts["Band"],
                    orientation="h",
                    marker=dict(
                        color=[COLOR_MAP_BAND.get(b, MUTED) for b in band_counts["Band"]],
                        opacity=0.85,
                        line=dict(width=0),
                    ),
                    text=band_counts["Count"],
                    textposition="outside",
                    textfont=dict(size=11, color="#94A3B8"),
                ))
                fig3.update_layout(**PLOTLY_THEME, height=200, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            section_header("Planning Observations", "Indicative spatial intelligence findings")
            insight_card(
                f"<strong>{n_crit} stations</strong> fall in the highest accessibility-risk band — "
                "clustered in high-demand corridors. Recommended for feasibility assessment.",
                "critical",
            )
            insight_card(
                f"<strong>{n_des} low-access zones</strong> show the weakest multimodal connectivity scores. "
                "Western expansion corridors appear most affected — pending ground validation.",
                "warning",
            )
            insight_card(
                f"<strong>{n_mclp} MCLP-modelled</strong> facility locations are estimated to maximise "
                "demand coverage within an 800m walk radius under current assumptions.",
            )


# ─────────────────────────────────────────────────────────────
# PAGE 3 — STATION INTELLIGENCE
# ─────────────────────────────────────────────────────────────

def page_station_intelligence(data: dict):
    page_title(
        "03 / Station Diagnostics",
        "Station Intelligence",
        "Per-station last-mile connectivity profile · LMCI decomposition · Demand-service alignment",
    )

    priority = data["priority_scores"]
    mismatch = data["mismatch"]

    if priority is None or priority.empty:
        notice("station_priority_scores.csv not found.")
        return

    station_names = sorted(priority["stop_name"].dropna().unique().tolist())

    col_sel, col_info = st.columns([2, 3], gap="medium")
    with col_sel:
        selected = st.selectbox(
            "Select Station",
            station_names,
            help="Choose a metro station to explore its connectivity profile",
            label_visibility="collapsed",
        )

    row = priority[priority["stop_name"] == selected].iloc[0]

    band     = str(row.get("priority_band", "Monitor"))
    score    = row.get("final_priority_score", None)
    rank     = row.get("final_priority_rank", None)
    intv     = str(row.get("recommended_intervention", "—"))
    mism     = str(row.get("mismatch_class", "—"))
    is_des   = str(row.get("is_persistent_desert", "")).lower() in ["true","1","yes"]
    mclp_sel = str(row.get("mclp_selected", "")).lower() in ["true","1","yes"]

    # ── STATION IDENTITY BANNER ───────────────────────────────
    bar_color = COLOR_MAP_BAND.get(band, MUTED)
    mclp_badge = "<span class='badge badge-low'>MCLP ✓</span>" if mclp_sel else ""
    desert_badge = "<span class='badge badge-critical'>DESERT ⚠</span>" if is_des else ""

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0D1420,#111827);border:1px solid #1A2535;"
        f"border-left:4px solid {bar_color};border-radius:0 12px 12px 0;padding:14px 20px;"
        f"margin-bottom:16px;display:flex;align-items:center;gap:16px'>"
        f"<div style='flex:1'>"
        f"<div style='color:#F1F5F9;font-size:1.2rem;font-weight:800;letter-spacing:-0.02em'>{selected}</div>"
        f"<div style='color:#475569;font-size:0.78rem;margin-top:3px'>{intv}</div>"
        f"</div>"
        f"<div style='display:flex;gap:10px;align-items:center'>"
        f"{badge_html(band)}"
        f"{mclp_badge}"
        f"{desert_badge}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI STRIP ─────────────────────────────────────────────
    kpi_cols = st.columns(5, gap="small")

    def _metric_card(col, label, value, fmt="{}", delta=None):
        try:
            display = fmt.format(float(value)) if value not in [None, "—"] and pd.notna(value) else "—"
        except Exception:
            display = str(value)
        with col:
            st.metric(label, display, delta=delta, border=True)

    _metric_card(kpi_cols[0], "Priority Score", score,                       "{:.1f}")
    _metric_card(kpi_cols[1], "Priority Rank",  rank,                        "#{:.0f}")
    _metric_card(kpi_cols[2], "Demand Signal",  row.get("demand_signal","—"),"{:.3f}")
    _metric_card(kpi_cols[3], "Temporal Gap",   row.get("temporal_gap","—"), "{:.3f}")
    kpi_cols[4].metric("MCLP Status", "✅ Selected" if mclp_sel else "❌ Not Selected", border=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── LMCI + RADAR ──────────────────────────────────────────
    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            section_header("LMCI Breakdown", "Last-Mile Connectivity Index by time window")
            lmci_cols = [c for c in ["Morning_LMCI","Midday_LMCI","Evening_LMCI"] if c in row.index]
            if lmci_cols:
                lmci_vals = [float(row.get(c, 0)) for c in lmci_cols]
                labels    = [c.replace("_LMCI","") for c in lmci_cols]
                colors    = [ACCENT, PURPLE, HIGH]
                fig = go.Figure()
                for i, (lbl, val, col_) in enumerate(zip(labels, lmci_vals, colors)):
                    fig.add_trace(go.Bar(
                        x=[lbl], y=[val],
                        marker=dict(color=col_, opacity=0.85, line=dict(width=0)),
                        text=[f"{val:.3f}"], textposition="outside",
                        textfont=dict(size=12, color="#CBD5E1"),
                        width=0.5, showlegend=False, name=lbl,
                    ))
                avg_lmci = np.mean(lmci_vals)
                fig.add_hline(
                    y=avg_lmci, line_dash="dot", line_color="#334155",
                    annotation_text=f"avg {avg_lmci:.3f}",
                    annotation_font_color="#475569", annotation_font_size=9,
                )
                fig.update_layout(
                    **{k: v for k, v in PLOTLY_THEME.items() if k != "yaxis"},
                    height=230,
                    showlegend=False,
                    barmode="group",
                    yaxis=dict(
                        **PLOTLY_THEME["yaxis"],
                        range=[0, max(lmci_vals or [1]) * 1.3 + 0.05]
                    ),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("LMCI time-window columns not found.")

    with right:
        with st.container(border=True):
            section_header("Connectivity Radar", "Normalised station attribute profile")
            radar_cols = {
                "demand_signal":                    "Demand Signal",
                "temporal_gap":                     "Temporal Gap",
                "equity_weighted_candidate_score":  "Equity Score",
            }
            available = {k: v for k, v in radar_cols.items() if k in row.index}
            if available:
                vals = [float(row.get(k, 0)) for k in available]
                lbls = list(available.values())
                vals_norm = [min(v, 1.0) if v <= 1.0 else v / 100.0 for v in vals]
                fig2 = go.Figure(go.Scatterpolar(
                    r=vals_norm + [vals_norm[0]],
                    theta=lbls + [lbls[0]],
                    fill="toself",
                    fillcolor="rgba(0,212,255,0.08)",
                    line=dict(color=ACCENT, width=2),
                    marker=dict(color=ACCENT, size=6),
                ))
                fig2.update_layout(
                    **{k: v for k, v in PLOTLY_THEME.items() if k not in ("xaxis","yaxis")},
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, color="#1E2D40", gridcolor="#1A2535",
                                        range=[0, 1], tickfont=dict(size=8, color="#334155")),
                        angularaxis=dict(color="#334155", tickfont=dict(size=10, color="#64748B")),
                    ),
                    height=230, showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("Insufficient signal columns for radar chart.")

    # ── STATION PROFILE — expandable ──────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        section_header("Full Station Profile", "Complete attribute intelligence record")
        display_cols = [c for c in [
            "stop_id", "stop_name", "stop_lat", "stop_lon",
            "mismatch_class", "desert_severity",
            "is_persistent_desert", "is_high_demand_low_service",
            "mclp_selection_rank", "recommended_intervention",
        ] if c in row.index]

        # Two-column attribute layout
        half = len(display_cols) // 2
        attr_l, attr_r = display_cols[:half], display_cols[half:]
        cl, cr = st.columns(2, gap="medium")
        for col_, attrs in [(cl, attr_l), (cr, attr_r)]:
            with col_:
                for a in attrs:
                    val = row.get(a, "—")
                    disp_val = str(val) if pd.notna(val) else "—"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"padding:7px 0;border-bottom:1px solid #0F1828'>"
                        f"<span style='color:#334155;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                        f"letter-spacing:0.06em'>{a.replace('_',' ').title()}</span>"
                        f"<span style='color:#CBD5E1;font-size:0.82rem;font-family:JetBrains Mono,monospace'>{disp_val}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────
# PAGE 4 — OPTIMIZATION & COVERAGE
# ─────────────────────────────────────────────────────────────

def page_optimization(data: dict):
    page_title(
        "04 / Optimization Engine",
        "Optimization & Coverage Analysis",
        "MCLP facility placement · Coverage curves · Marginal demand attribution · Equity-weighted scoring",
    )

    coverage  = data["mclp_coverage"]
    selected  = data["mclp_selected"]
    candidates = data["mclp_candidates"]

    # ── EXECUTIVE INSIGHT ─────────────────────────────────────
    if coverage is not None and "coverage_pct" in coverage.columns:
        k5_cov = coverage[coverage.get("k", coverage.iloc[:,0]) == 5]["coverage_pct"].values
        cov_msg = f"<strong>{k5_cov[0]:.1f}%</strong> demand coverage achieved at k=5 facilities." if len(k5_cov) else ""
        if cov_msg:
            insight_card(
                f"{cov_msg} Each additional facility beyond k=5 yields diminishing marginal returns — "
                "representing the optimal deployment threshold.",
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈  Coverage Curve", "🏆  Selected Stations", "📋  Candidate Rankings"])

    with tab1:
        with st.container(border=True):
            section_header("MCLP Coverage Curve", "Cumulative demand coverage (%) as a function of facility count k")
            if coverage is not None and not coverage.empty and "coverage_pct" in coverage.columns:
                k_col = "k" if "k" in coverage.columns else coverage.columns[0]
                coverage_plot = coverage.copy()

                fig = go.Figure()
                # Fill area
                fig.add_trace(go.Scatter(
                    x=coverage_plot[k_col], y=coverage_plot["coverage_pct"],
                    mode="none", fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.05)",
                    showlegend=False,
                ))
                # Line
                fig.add_trace(go.Scatter(
                    x=coverage_plot[k_col], y=coverage_plot["coverage_pct"],
                    mode="lines+markers",
                    line=dict(color=ACCENT, width=2.5),
                    marker=dict(size=8, color=ACCENT,
                                line=dict(color="#0A0E1A", width=2)),
                    name="Coverage %",
                    hovertemplate="k=%{x}<br>Coverage: %{y:.1f}%<extra></extra>",
                ))
                fig.add_hline(y=80, line_dash="dash", line_color=LOW, line_width=1,
                              annotation_text="80% target", annotation_font_color=LOW,
                              annotation_font_size=9)
                # Mark k=5
                if 5 in coverage_plot[k_col].values:
                    k5_val = coverage_plot[coverage_plot[k_col]==5]["coverage_pct"].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[5], y=[k5_val], mode="markers",
                        marker=dict(size=14, color=ACCENT, symbol="diamond",
                                    line=dict(color="#0A0E1A", width=2)),
                        name=f"k=5 ({k5_val:.1f}%)",
                        hovertemplate=f"Optimal k=5<br>Coverage: {k5_val:.1f}%<extra></extra>",
                    ))
                fig.update_layout(
                    **{k: v for k, v in PLOTLY_THEME.items() if k not in ("xaxis", "yaxis")},
                    height=340,
                    xaxis=dict(**PLOTLY_THEME["xaxis"], title="Number of Facilities (k)", dtick=1),
                    yaxis=dict(**PLOTLY_THEME["yaxis"], title="Coverage (%)", range=[0, 105]),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Marginal gain bar
                if len(coverage_plot) > 1:
                    coverage_plot = coverage_plot.copy()
                    coverage_plot["marginal_gain"] = coverage_plot["coverage_pct"].diff().fillna(
                        coverage_plot["coverage_pct"].iloc[0]
                    )
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    section_header("Marginal Coverage Gain", "Additional coverage per incremental facility")
                    fig_mg = go.Figure(go.Bar(
                        x=coverage_plot[k_col],
                        y=coverage_plot["marginal_gain"],
                        marker=dict(
                            color=coverage_plot["marginal_gain"],
                            colorscale=[[0,"#1E2D40"],[1,ACCENT]],
                            line=dict(width=0),
                        ),
                        text=coverage_plot["marginal_gain"].map(lambda x: f"+{x:.1f}%"),
                        textposition="outside",
                        textfont=dict(size=10, color="#64748B"),
                        width=0.6,
                    ))
                    fig_mg.update_layout(
                        **{k: v for k, v in PLOTLY_THEME.items() if k not in ("xaxis", "yaxis")},
                        height=200,
                        showlegend=False,
                        xaxis=dict(**PLOTLY_THEME["xaxis"], title="k"),
                        yaxis=dict(**PLOTLY_THEME["yaxis"], title="Marginal Gain (%)"),
                    )
                    st.plotly_chart(fig_mg, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("mclp_coverage_by_k.csv not found.")

    with tab2:
        with st.container(border=True):
            section_header("MCLP Selected Stations", "Optimally placed facilities ranked by marginal demand coverage")
            if selected is not None and not selected.empty:
                if "marginal_weighted_demand" in selected.columns and "stop_name" in selected.columns:
                    plot_df = selected.sort_values("marginal_weighted_demand", ascending=True).tail(15)
                    fig3 = go.Figure(go.Bar(
                        x=plot_df["marginal_weighted_demand"],
                        y=plot_df["stop_name"],
                        orientation="h",
                        marker=dict(
                            color=plot_df["marginal_weighted_demand"],
                            colorscale=[[0,"#1A2535"],[0.5,PURPLE],[1,ACCENT]],
                            line=dict(width=0),
                        ),
                        text=plot_df["marginal_weighted_demand"].map(lambda x: f"{x:.0f}"),
                        textposition="outside",
                        textfont=dict(size=10, color="#64748B"),
                    ))
                    fig3.update_layout(**plotly_layout(
                        height=380, showlegend=False,
                        xaxis=axis_layout("xaxis", title="Marginal Weighted Demand"),
                        yaxis=axis_layout("yaxis", tickfont=dict(size=10, color="#94A3B8")),
                    ))
                    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

                # Ranked cards
                show_sel = selected.copy()
                if "marginal_weighted_demand" in show_sel.columns:
                    show_sel = show_sel.sort_values("marginal_weighted_demand", ascending=False).head(10)
                max_val = show_sel["marginal_weighted_demand"].max() if "marginal_weighted_demand" in show_sel.columns else 1
                cards_html = ""
                for i, (_, srow) in enumerate(show_sel.iterrows()):
                    name  = str(srow.get("stop_name", "—"))
                    val   = float(srow.get("marginal_weighted_demand", 0))
                    pct   = max(4, int((val / max_val) * 100)) if max_val else 4
                    cards_html += rank_card_html(i+1, name, val, max_val)
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                notice("mclp_selected_stations.csv not found.")

    with tab3:
        with st.container(border=True):
            section_header("Candidate Station Rankings", "Top 20 stations ranked by MCLP potential")
            if candidates is not None and not candidates.empty:
                rank_col = "candidate_rank" if "candidate_rank" in candidates.columns else None
                top_cands = candidates.nsmallest(20, rank_col) if rank_col else candidates.head(20)
                show_cols = [c for c in [
                    "stop_name", "candidate_rank", "covered_demand_points_if_selected",
                    "weighted_demand_if_selected", "equity_weighted_candidate_score",
                ] if c in top_cands.columns]
                disp2 = top_cands[show_cols].rename(columns={
                    "stop_name": "Station", "candidate_rank": "Rank",
                    "covered_demand_points_if_selected": "Demand Points",
                    "weighted_demand_if_selected": "Weighted Demand",
                    "equity_weighted_candidate_score": "Equity Score",
                })
                if "Rank" in disp2.columns:
                    disp2["Rank"] = disp2["Rank"].map(lambda x: f"#{int(x)}" if pd.notna(x) else "—")
                st.dataframe(disp2, hide_index=True, use_container_width=True)
            else:
                notice("mclp_candidate_scores.csv not found.")


# ─────────────────────────────────────────────────────────────
# PAGE 5 — SCENARIO SIMULATION
# ─────────────────────────────────────────────────────────────

def page_simulation(data: dict):
    page_title(
        "05 / Scenario Modelling",
        "Scenario Simulation Engine",
        "Intervention impact modelling · Accessibility improvement potential · Connectivity uplift comparison",
    )

    impacts   = data["sim_impacts"]
    ranking   = data["sim_ranking"]
    network   = data["sim_network"]
    scenarios = data["sim_scenarios"]

    # ── NETWORK SUMMARY KPIS ──────────────────────────────────
    if network is not None and not network.empty:
        with st.container(horizontal=True):
            st.metric("Accessibility Uplift Score",  _fmt_num(_safe_val(network,"estimated_daily_ridership_gain")),
                      "Composite index · Not a count", border=True)
            st.metric("Mean LMCI Enhancement",       f"{_safe_val(network,'mean_lmci_gain'):.3f}",
                      "Modelled connectivity gain", border=True)
            st.metric("Scenarios Evaluated",          _fmt_num(_safe_val(network,"top_interventions_evaluated")),
                      "Intervention combinations", border=True)
            st.metric("Low-Access Zones Targeted",    _fmt_num(_safe_val(network,"persistent_deserts_targeted")),
                      "Candidate recovery areas", border=True)
            st.metric("Multimodal Scenarios",         _fmt_num(_safe_val(network,"multimodal_projects_recommended")),
                      "Hub integration candidates", border=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Network impact narrative
        gain    = _safe_val(network,"estimated_daily_ridership_gain", 0)
        lm_gain = _safe_val(network,"mean_lmci_gain", 0)
        st.markdown(
            f"<div class='network-impact'>"
            f"<div class='network-impact-label'>Indicative Network Impact Assessment — Scenario-Based Modelling</div>"
            f"<div class='network-impact-text'>Based on accessibility and multimodal connectivity modelling, "
            f"the recommended interventions are estimated to yield a "
            f"<strong style='color:{ACCENT}'>relative Accessibility Uplift Score of {_fmt_num(gain)}</strong> "
            f"<em style='color:#475569;font-size:0.8em'>(composite index — not a ridership count)</em>, "
            f"with a mean LMCI improvement of "
            f"<strong style='color:{LOW}'>+{lm_gain:.3f} points</strong> across the network. "
            f"Multimodal hub scenarios show higher relative connectivity enhancement "
            f"compared to single-mode feeder scenarios across modelled conditions. "
            f"<em style='color:#334155;font-size:0.78rem'>Indicative simulation only. Not a ridership forecast. Requires feasibility validation.</em>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── FILTER ────────────────────────────────────────────────
    if scenarios is not None and "intervention_type" in scenarios.columns:
        types = ["All"] + sorted(scenarios["intervention_type"].dropna().unique().tolist())
        sel_type = st.selectbox("Filter by Intervention Type", types, label_visibility="collapsed")
        filtered = scenarios if sel_type == "All" else scenarios[scenarios["intervention_type"] == sel_type]
    else:
        filtered = scenarios
        sel_type = "All"

    tab1, tab2, tab3 = st.tabs(["🏆  Station Impact Rankings", "⚡  Intervention Comparison", "📋  Scenario Matrix"])

    with tab1:
        with st.container(border=True):
            section_header("Top Station Interventions", "Ranked by modelled accessibility uplift potential — scenario-based estimates")
            if impacts is not None and not impacts.empty:
                top_imp = (
                    impacts.nlargest(15, "simulation_priority_score")
                    if "simulation_priority_score" in impacts.columns
                    else impacts.head(15)
                )

                # Scenario cards
                if "scenario_name" in top_imp.columns and "simulated_daily_ridership_gain" in top_imp.columns:
                    max_rid = top_imp["simulated_daily_ridership_gain"].max()
                    max_lmci = top_imp["lmci_gain"].max() if "lmci_gain" in top_imp.columns else 1
                    for _, irow in top_imp.iterrows():
                        name    = str(irow.get("stop_name","—"))
                        sc_name = str(irow.get("scenario_name","—"))
                        rid     = float(irow.get("simulated_daily_ridership_gain",0))
                        lmci    = float(irow.get("lmci_gain",0)) if "lmci_gain" in irow.index else 0
                        net_val = float(irow.get("network_value_score",0)) if "network_value_score" in irow.index else 0
                        cost    = str(irow.get("cost_band","—"))
                        rid_pct = max(4, int((rid / max_rid)*100)) if max_rid else 4
                        cost_col = {"Low": LOW, "Medium": MEDIUM, "High": HIGH, "Very High": CRITICAL}.get(cost, MUTED)
                        st.markdown(
                            f"<div class='scenario-card'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:start'>"
                            f"<div><div class='scenario-type'>{sc_name}</div>"
                            f"<div class='scenario-name'>{name}</div></div>"
                            f"<span class='badge' style='background:rgba(0,0,0,0.2);color:{cost_col};"
                            f"border:1px solid {cost_col}44'>{cost} Cost</span>"
                            f"</div>"
                            f"<div class='scenario-metrics'>"
                            f"<div><div class='scenario-metric-val' style='color:{ACCENT}'>{_fmt_num(rid)}</div>"
                            f"<div class='scenario-metric-lbl'>Impact Index</div></div>"
                            f"<div><div class='scenario-metric-val' style='color:{LOW}'>+{lmci:.3f}</div>"
                            f"<div class='scenario-metric-lbl'>LMCI Uplift</div></div>"
                            f"<div><div class='scenario-metric-val' style='color:{PURPLE}'>{net_val:.2f}</div>"
                            f"<div class='scenario-metric-lbl'>Network Value</div></div>"
                            f"</div>"
                            f"<div class='rank-bar' style='margin-top:10px'>"
                            f"<div class='rank-bar-fill' style='width:{rid_pct}%'></div></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Ridership chart
                if "simulated_daily_ridership_gain" in impacts.columns:
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    section_header("Accessibility Impact Index by Station", "Top 12 interventions · Relative connectivity benefit score")
                    top15 = impacts.nlargest(12, "simulated_daily_ridership_gain")
                    fig = go.Figure(go.Bar(
                        x=top15["simulated_daily_ridership_gain"],
                        y=top15["stop_name"],
                        orientation="h",
                        marker=dict(
                            color=top15["simulated_daily_ridership_gain"],
                            colorscale=[[0,"#1A2535"],[0.5,PURPLE],[1,ACCENT]],
                            line=dict(width=0),
                        ),
                        text=top15["simulated_daily_ridership_gain"].map(lambda x: f"+{_fmt_num(x)}"),
                        textposition="outside",
                        textfont=dict(size=10, color="#64748B"),
                    ))
                    fig.update_layout(**plotly_layout(
                        height=380, showlegend=False,
                        xaxis=axis_layout("xaxis", title="Accessibility Impact Index (Relative)"),
                        yaxis=axis_layout("yaxis", tickfont=dict(size=10, color="#94A3B8")),
                    ))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("simulation_station_impacts.csv not found.")

    with tab2:
        with st.container(border=True):
            section_header("Intervention Type Comparison", "Relative modelled performance by intervention category — indicative planning reference")
            if ranking is not None and not ranking.empty:
                metric_opts = [c for c in [
                    "simulation_priority_score","lmci_gain",
                    "simulated_daily_ridership_gain","network_value_score",
                ] if c in ranking.columns]
                if metric_opts:
                    chosen = st.selectbox(
                        "Metric", metric_opts,
                        format_func=lambda x: x.replace("_"," ").title(),
                        label_visibility="collapsed",
                    )
                    rank_df = ranking.sort_values(chosen, ascending=False).copy()
                    cost_col_map = {"Low": LOW, "Medium": MEDIUM, "High": HIGH, "Very High": CRITICAL}
                    bar_colors = [
                        cost_col_map.get(str(c), MUTED)
                        for c in rank_df.get("cost_band", pd.Series())
                    ] if "cost_band" in rank_df.columns else [ACCENT] * len(rank_df)

                    name_col = "scenario_name" if "scenario_name" in rank_df.columns else rank_df.columns[0]
                    fig2 = go.Figure(go.Bar(
                        x=rank_df[name_col],
                        y=rank_df[chosen],
                        marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
                        text=rank_df[chosen].map(lambda x: f"{x:.2f}"),
                        textposition="outside",
                        textfont=dict(size=11, color="#94A3B8"),
                        width=0.55,
                    ))
                    fig2.update_layout(**plotly_layout(
                        height=320,
                        xaxis=axis_layout("xaxis", title="Scenario Type"),
                        yaxis=axis_layout("yaxis", title=chosen.replace("_", " ").title()),
                    ))
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

                    # Strategic ranking callouts
                    if not rank_df.empty:
                        best = rank_df.iloc[0]
                        best_name = str(best.get(name_col, "—"))
                        best_val  = float(best.get(chosen, 0))
                        insight_card(
                            f"<strong>{best_name}</strong> shows the highest relative modelled performance "
                            f"on <em>{chosen.replace('_',' ')}</em> ({best_val:.2f}). "
                            "This category may warrant priority consideration in feasibility planning — subject to ground-level validation.",
                        )
                else:
                    st.dataframe(ranking, hide_index=True, use_container_width=True)
            else:
                notice("simulation_intervention_ranking.csv not found.")

    with tab3:
        with st.container(border=True):
            section_header("Full Scenario Matrix", "All station × intervention combinations · Indicative accessibility estimates")
            if filtered is not None and not filtered.empty:
                show_cols = [c for c in [
                    "stop_name","scenario_name","intervention_type",
                    "lmci_gain","simulated_daily_ridership_gain",
                    "network_value_score","simulation_priority_score","cost_band",
                ] if c in filtered.columns]
                disp = filtered[show_cols].rename(columns={
                    "stop_name":"Station","scenario_name":"Scenario",
                    "intervention_type":"Type","lmci_gain":"LMCI Δ",
                    "simulated_daily_ridership_gain":"Impact Index",
                    "network_value_score":"Network Value",
                    "simulation_priority_score":"Priority","cost_band":"Cost",
                })
                if "LMCI Δ" in disp.columns:
                    disp["LMCI Δ"] = disp["LMCI Δ"].map(lambda x: f"+{float(x):.4f}" if pd.notna(x) else "—")
                if "Impact Index" in disp.columns:
                    disp["Impact Index"] = disp["Impact Index"].map(lambda x: f"{_fmt_num(x)}" if pd.notna(x) else "—")
                st.dataframe(disp, hide_index=True, use_container_width=True)
            else:
                notice("Scenario data unavailable.")


# ─────────────────────────────────────────────────────────────
# PAGE 6 — RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────────────────────

def page_recommendations(data: dict):
    page_title(
        "06 / Planning Support",
        "Intervention Planning Engine",
        "Indicative action priorities · Scenario-based planning tiers · Accessibility investment guidance",
    )

    insights = data["insights_top5"]
    priority = data["priority_scores"]
    sim_net  = data["sim_network"]

    # ── STRATEGIC KPIs ────────────────────────────────────────
    if priority is not None:
        n_critical = int((priority.get("priority_band","") == "Critical").sum()) if "priority_band" in priority.columns else 0
        n_high     = int((priority.get("priority_band","") == "High").sum()) if "priority_band" in priority.columns else 0
        n_deserts  = int(priority.get("is_persistent_desert", pd.Series(False)).astype(str).str.lower().isin(["true","1","yes"]).sum()) if "is_persistent_desert" in priority.columns else 0

        with st.container(horizontal=True):
            st.metric("Highest-Priority Stations", n_critical, "Recommended for assessment", border=True)
            st.metric("Elevated-Priority Stations", n_high,    "Near-term review window", border=True)
            st.metric("Low-Access Recovery Zones", n_deserts,  "Candidate intervention areas", border=True)
            if sim_net is not None and not sim_net.empty:
                gain     = _safe_val(sim_net,"estimated_daily_ridership_gain",0)
                lmci_g   = _safe_val(sim_net,"mean_lmci_gain",0)
                st.metric("Accessibility Uplift Score", _fmt_num(gain), "Composite index · Not a count", border=True)
                st.metric("Mean LMCI Enhancement", f"+{lmci_g:.3f}", "Modelled connectivity gain", border=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── EXECUTIVE INSIGHT PANELS ──────────────────────────────
    icols = st.columns(3, gap="small")
    with icols[0]:
        n_c = n_critical if priority is not None else "?"
        insight_card(
            f"<strong>{n_c} stations</strong> score in the highest accessibility-risk band. "
            "Feeder or multimodal interventions are indicated — feasibility assessment recommended.",
            "critical",
        )
    with icols[1]:
        insight_card(
            "Integrated <strong>multimodal hubs</strong> show the highest relative accessibility uplift — "
            "outperforming feeder-only scenarios on connectivity enhancement across modelled conditions.",
            "warning",
        )
    with icols[2]:
        insight_card(
            "Low-access zones remain concentrated in western expansion corridors. "
            "These areas are <strong>candidate sites</strong> for Phase 2 connectivity investment.",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── DEPLOY NOW ────────────────────────────────────────────
    with st.container(border=True):
        section_header("📋 Priority Assessment List", "Stations flagged for near-term feasibility review · Indicative planning priority")

        deploy_df = None
        if insights is not None and not insights.empty:
            if "deploy_action" in insights.columns:
                deploy_df = insights[insights["deploy_action"].str.upper() == "DEPLOY NOW"]
            else:
                deploy_df = insights
        elif priority is not None and "priority_band" in priority.columns:
            deploy_df = priority[priority["priority_band"] == "Critical"].nlargest(5,"final_priority_score")

        if deploy_df is not None and not deploy_df.empty:
            for _, row in deploy_df.iterrows():
                name  = row.get("stop_name","Unknown")
                score = row.get("final_priority_score","—")
                intv  = row.get("recommended_intervention","Feasibility assessment indicated")
                band  = str(row.get("priority_band","Critical"))
                score_str = f"{float(score):.1f}" if score != "—" and pd.notna(score) else "—"
                st.markdown(
                    f"<div class='deploy-card'>"
                    f"<div><div class='deploy-name'>{name}</div>"
                    f"<div class='deploy-action'>{intv}</div></div>"
                    f"<div style='display:flex;align-items:center;gap:10px'>"
                    f"{badge_html(band)}"
                    f"<div class='deploy-score'>Score {score_str}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            notice("No stations in the highest planning tier. Review priority thresholds.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── MONITOR + INTERVENTION DISTRIBUTION ──────────────────
    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True):
            section_header("Monitoring Watchlist", "Elevated-priority stations · Near-term planning review")
            if priority is not None and "priority_band" in priority.columns:
                monitor_df = priority[priority["priority_band"].isin(["High","Medium"])].nlargest(8,"final_priority_score")
                max_s = monitor_df["final_priority_score"].max() if "final_priority_score" in monitor_df.columns else 1
                for _, mrow in monitor_df.iterrows():
                    name  = str(mrow.get("stop_name","—"))
                    score = float(mrow.get("final_priority_score",0))
                    band  = str(mrow.get("priority_band","Medium"))
                    pct   = max(4, int((score/max_s)*100)) if max_s else 4
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:12px;padding:8px 0;"
                        f"border-bottom:1px solid #0F1828'>"
                        f"<div style='flex:1'>"
                        f"<div style='color:#CBD5E1;font-size:0.82rem;font-weight:600'>{name}</div>"
                        f"<div class='rank-bar'><div class='rank-bar-fill' style='width:{pct}%;"
                        f"background:{COLOR_MAP_BAND.get(band,MUTED)}'></div></div>"
                        f"</div>"
                        f"{badge_html(band)}"
                        f"<span style='color:#334155;font-size:0.72rem;font-family:JetBrains Mono,monospace;min-width:32px'>{score:.1f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                notice("Priority data unavailable.")

    with right:
        with st.container(border=True):
            section_header("Indicative Action Distribution", "Planning action categories by station count")
            if priority is not None and "recommended_intervention" in priority.columns:
                intv_counts = priority["recommended_intervention"].value_counts().head(8).reset_index()
                intv_counts.columns = ["Intervention","Count"]
                fig = go.Figure(go.Bar(
                    x=intv_counts.sort_values("Count")["Count"],
                    y=intv_counts.sort_values("Count")["Intervention"],
                    orientation="h",
                    marker=dict(
                        color=intv_counts.sort_values("Count")["Count"],
                        colorscale=[[0,"#1A2535"],[1,ACCENT]],
                        line=dict(width=0),
                    ),
                    text=intv_counts.sort_values("Count")["Count"],
                    textposition="outside",
                    textfont=dict(size=11, color="#64748B"),
                ))
                fig.update_layout(**plotly_layout(
                    height=280, showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=axis_layout("yaxis", tickfont=dict(size=9, color="#94A3B8")),
                ))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                notice("Intervention data unavailable.")

    # ── NETWORK IMPACT NARRATIVE ──────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if sim_net is not None and not sim_net.empty:
        gain    = _safe_val(sim_net,"estimated_daily_ridership_gain",0)
        lmci_g  = _safe_val(sim_net,"mean_lmci_gain",0)
        with st.container(border=True):
            section_header("Network-Level Planning Summary", "Scenario-based assessment · Indicative investment sequencing")
            st.markdown(
                f"<div class='network-impact' style='margin-top:8px'>"
                f"<div class='network-impact-label'>Indicative Network Assessment — Scenario-Based Modelling</div>"
                f"<div class='network-impact-text'>"
                f"Based on accessibility and multimodal connectivity modelling, the recommended intervention portfolio "
                f"is estimated to yield a <strong style='color:{ACCENT}'>relative Accessibility Uplift Score of {_fmt_num(gain)}</strong> "
                f"<em style='color:#475569;font-size:0.8em'>(composite index — not a ridership count)</em>, "
                f"with a mean LMCI enhancement of "
                f"<strong style='color:{LOW}'>+{lmci_g:.3f} points</strong> across the Hyderabad Metro network. "
                f"Phase 1 planning should concentrate on highest-priority stations with multimodal upgrade potential. "
                f"Phase 2 should address low-access zones through feeder route densification in expansion corridors. "
                f"<em style='color:#334155;font-size:0.78rem'>Indicative simulation only. All outputs reflect accessibility and connectivity assumptions — not AFC, passenger count, or revenue data. Ground-level feasibility validation required before operational use.</em>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── EXPORT ────────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if priority is not None:
        st.download_button(
            "⬇  Export Accessibility Priority Report (CSV)",
            priority.to_csv(index=False).encode("utf-8"),
            file_name="hyd_metro_accessibility_priority.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    data = load_all()
    page = render_sidebar()

    if page == "Executive Overview":
        page_executive_overview(data)
    elif page == "Transit Intelligence Map":
        page_transit_map(data)
    elif page == "Station Intelligence":
        page_station_intelligence(data)
    elif page == "Optimization & Coverage":
        page_optimization(data)
    elif page == "Scenario Simulation":
        page_simulation(data)
    elif page == "Intervention Planning":
        page_recommendations(data)


if __name__ == "__main__":
    main()
