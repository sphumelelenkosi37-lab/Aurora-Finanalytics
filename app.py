"""AURORA FINANALYTICS — live financial-news sentiment vs. price action.

Pulls live Yahoo Finance headlines for a ticker, scores them with FinBERT
(a financial-sentiment transformer), overlays the scores on hourly price
action, and tests whether daily sentiment and daily returns co-move —
same-day and next-day. Built as a research-grade personal project.

Honesty rules this app follows
------------------------------
1. Every panel states whether it is showing LIVE data or the offline SAMPLE.
2. If a headline cannot be placed on the price line without inventing a
   timestamp, it is *not* silently snapped to an edge bar — it is reported
   in a "not priced yet" list and excluded from the statistics.
3. Every statistic is shown with its sample size, and with a caveat when the
   sample is too small to mean anything.

Visual theme: deep navy, subtly textured background, gold/bronze metallic
accents. Layout: logo (top-left) -> centred nav tabs -> centred gold-outlined
CTA -> instruction line -> metallic-framed output panel.
"""

from datetime import datetime, timedelta, timezone
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:  # optional: only used for p-values / t-tests
    from scipy import stats as _scipy_stats

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Sentiment model. FinBERT: BERT-base fine-tuned on the Financial PhraseBank.
MODEL_ID = "ProsusAI/finbert"
# Exchange timezone used to define a "trading day" (and therefore a daily return).
MARKET_TZ = "America/New_York"
# A headline is drawn on the price line only if a bar sits within this gap.
MARKER_TOLERANCE = pd.Timedelta(hours=4)
# Trading hours used by the event study: average drift over N hourly bars.
EVENT_HORIZON_BARS = 4


# ----------------------------------------------------------------------------
# Environment: quiet, predictable behaviour
# ----------------------------------------------------------------------------
import html as _html
import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# FinBERT is a public model, so no token is required. Setting HF_TOKEN raises
# the Hugging Face rate limit and removes the "unauthenticated requests" notice
# from the logs; without it, downloads still work, so we simply keep the log
# clean and surface any real failure to the user instead.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Brand mark used for the browser tab: an inline SVG, so no extra asset file
# is needed and the app stays a single-file deployment.
PAGE_ICON = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAzMiAzMic+PHJlY3QgeD0nMS41JyB5PScxLjUnIHdpZHRoPScyOScgaGVpZ2h0PScyOScgcng9JzYnIGZpbGw9JyMwYTExMjInIHN0cm9rZT0nI2M4OTY0YScgc3Ryb2tlLXdpZHRoPScxLjYnLz48cmVjdCB4PSc4LjUnIHk9JzE1JyB3aWR0aD0nMy40JyBoZWlnaHQ9JzguNScgZmlsbD0nIzhiOWFiNScvPjxyZWN0IHg9JzE0LjMnIHk9JzknIHdpZHRoPSczLjQnIGhlaWdodD0nMTQuNScgZmlsbD0nI2M4OTY0YScvPjxyZWN0IHg9JzIwLjEnIHk9JzEyJyB3aWR0aD0nMy40JyBoZWlnaHQ9JzExLjUnIGZpbGw9JyNlOGM4N2EnLz48L3N2Zz4="
)

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AURORA FINANALYTICS",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
# Subtle fractal-noise texture (percent-encoded so no HTML parser surprises).
NOISE = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E"
    "%3Cfilter id='n'%3E"
    "%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4'"
    " stitchTiles='stitch'/%3E"
    "%3CfeColorMatrix type='saturate' values='0'/%3E"
    "%3C/filter%3E"
    "%3Crect width='160' height='160' filter='url(%23n)' opacity='0.05'/%3E"
    "%3C/svg%3E"
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --navy-950: #060a14;
        --navy-900: #080d1a;
        --navy-800: #0a1122;
        --navy-700: #0e1730;
        --gold:        #c8964a;
        --gold-bright: #e8c87a;
        --gold-dim:    rgba(200, 150, 74, 0.28);
        --gold-line:   rgba(200, 150, 74, 0.18);
        --text:        #e9eef7;
        --muted:       #8b9ab5;
    }}

    /* ---------- canvas: navy gradient + faint grain ---------- */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: var(--navy-900);
        background-image:
            url("{NOISE}"),
            radial-gradient(1200px 620px at 12% -12%, #17253f 0%, rgba(23,37,63,0) 62%),
            radial-gradient(1000px 560px at 88% 112%, #101d38 0%, rgba(16,29,56,0) 58%),
            linear-gradient(162deg, #0a1224 0%, #0b1428 42%, var(--navy-950) 100%);
        background-attachment: fixed, fixed, fixed, fixed;
        background-repeat: repeat, no-repeat, no-repeat, no-repeat;
        color: var(--text);
    }}

    /* Streamlit chrome: transparent header, keep the Deploy control visible */
    [data-testid="stHeader"] {{
        background: transparent;
        border-bottom: none;
    }}
    [data-testid="stToolbar"] {{ right: 1.2rem; }}
    [data-testid="stAppDeployButton"] button {{
        background: linear-gradient(180deg, rgba(200,150,74,0.14), rgba(200,150,74,0.05));
        color: var(--gold-bright);
        border: 1px solid rgba(200,150,74,0.55);
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}
    [data-testid="stAppDeployButton"] button:hover {{
        border-color: var(--gold-bright);
        color: #fff;
    }}
    [data-testid="stToolbar"] svg {{ fill: var(--muted); }}
    [data-testid="stMainMenu"]:hover svg {{ fill: var(--gold-bright); }}

    #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; }}
    [data-testid="stDecoration"] {{ display: none; }}

    /* ---------- typography ---------- */
    /* The :not(...) chain is load-bearing. Streamlit draws its own chrome
       icons (the sidebar collapse control, expander chevrons, spinners, the
       dataframe toolbar) as Material Symbols *ligatures*: the element's text
       content is the icon NAME — e.g. "keyboard_double_arrow_left" — and only
       the icon font turns that name into a glyph. A bare
       `[data-testid="stAppViewContainer"] *` selector also hits those spans,
       so the name gets painted as literal body text and the control reads
       "keyboard_double_arrow_left" instead of showing an arrow. Icon elements
       are excluded from the font override here and restored explicitly in the
       icon-font guard further down. */
    html, body,
    [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]) {{
        font-family: "Source Sans Pro", "Segoe UI", system-ui, sans-serif;
    }}
    h1, h2, h3, h4 {{ color: var(--text); letter-spacing: 0.02em; font-weight: 600; }}
    p, span, label, div {{ color: var(--text); }}
    a {{ color: var(--gold-bright); }}

    /* ---------- sidebar: terminal controls ---------- */
    [data-testid="stSidebar"] {{
        background-color: transparent;
        background-image: linear-gradient(180deg, #0c1628 0%, #070d1b 100%);
        border-right: 1px solid var(--gold-line);
        box-shadow: inset -1px 0 0 rgba(255,255,255,0.02);
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.4rem; }}
    [data-testid="stSidebar"] h3 {{
        color: var(--gold-bright);
        font-size: 0.95rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }}
    [data-testid="stSidebar"] h3::after {{
        content: "";
        display: block;
        width: 44px;
        height: 1px;
        margin-top: 0.55rem;
        background: linear-gradient(90deg, var(--gold) 0%, rgba(200,150,74,0) 100%);
    }}
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] .stSlider label p {{
        color: var(--muted) !important;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* ---------- inputs: machined metal fields ---------- */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    [data-testid="stTextInputRootElement"] {{
        background: rgba(255,255,255,0.035) !important;
        border-color: rgba(200,150,74,0.30) !important;
        border-radius: 7px !important;
    }}
    .stTextInput input,
    div[data-baseweb="input"] input {{
        background: transparent !important;
        color: var(--text) !important;
        font-weight: 500;
        letter-spacing: 0.04em;
    }}
    div[data-baseweb="input"]:focus-within,
    [data-testid="stTextInputRootElement"]:focus-within {{
        border-color: var(--gold-bright) !important;
        box-shadow: 0 0 0 2px rgba(200,150,74,0.16) !important;
    }}

    /* slider — thumb is the element wrapping the hidden range input */
    [data-testid="stSlider"] div:has(> div > input[type="range"]),
    [data-testid="stSlider"] div[style*="translate(-50%, -50%)"] {{
        background: linear-gradient(180deg, var(--gold-bright), var(--gold)) !important;
        width: 14px !important;
        height: 14px !important;
        border: 1px solid #6b4f1d !important;
        border-radius: 50% !important;
        box-shadow: 0 0 10px rgba(200,150,74,0.45) !important;
    }}
    /* the track sits immediately before the thumb; the fill % is baked into a
       per-value emotion class, so recolour with a filter instead of replacing
       the gradient (this preserves the dynamic fill boundary). */
    [data-testid="stSlider"] [role="group"] > div > div:first-child {{
        filter: hue-rotate(36deg) saturate(0.5) brightness(0.83);
        height: 4px !important;
        border-radius: 2px;
    }}
    [data-testid="stSlider"] [data-testid="stSliderThumbValue"] p,
    [data-testid="stSlider"] [data-testid="stSliderTickBar"] p {{
        color: var(--muted) !important;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
    }}
    [data-testid="stSlider"] [data-testid="stSliderThumbValue"] p {{
        color: var(--gold-bright) !important;
    }}

    /* ---------- top nav (text tabs) ---------- */
    div[class*="st-key-nav_"] button {{
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: var(--muted) !important;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        padding: 0.35rem 0.15rem 0.45rem 0.15rem !important;
        min-height: 0 !important;
        transition: color 120ms ease, border-color 120ms ease;
    }}
    div[class*="st-key-nav_"] button p {{
        white-space: nowrap;
        font-size: 0.85rem;
        color: var(--muted) !important;
    }}
    div[class*="st-key-nav_"] button:hover {{
        color: var(--gold-bright) !important;
        border-bottom-color: rgba(200,150,74,0.45) !important;
    }}
    div[class*="st-key-nav_"] button:hover p {{
        color: var(--gold-bright) !important;
    }}
    div[class*="st-key-nav_"] button[kind="primary"],
    div[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] {{
        color: var(--gold-bright) !important;
        border-bottom-color: var(--gold) !important;
    }}
    div[class*="st-key-nav_"] button[kind="primary"] p,
    div[class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p {{
        color: var(--gold-bright) !important;
        font-weight: 600;
    }}

    /* ---------- nav row: four tabs, one line ---------- */
    /* The sidebar leaves a narrow main column on small windows; without this the
       fourth tab wraps to a second row and the "centred tab bar" reads as broken. */
    div[class*="st-key-nav_bar"] {{
        flex-wrap: nowrap !important;
        gap: 0.05rem !important;
        overflow-x: auto !important;
        scrollbar-width: none;
    }}
    div[class*="st-key-nav_bar"]::-webkit-scrollbar {{ display: none; }}
    div[class*="st-key-nav_bar"] > div[class*="st-key-nav_"] {{ flex: 0 0 auto !important; }}
    div[class*="st-key-nav_bar"] button {{ padding: 0.35rem 0.22rem 0.45rem 0.22rem !important; }}

    /* ---------- primary call-to-action ---------- */
    /* children of a horizontal container are flex: 0 0 fit-content, so the
       width has to be set through flex, not width. */
    div[class*="st-key-run_pipeline"] {{
        flex: 0 0 26rem !important;
        max-width: 100%;
    }}
    div[class*="st-key-run_pipeline"] button {{
        width: 100%;
        background: linear-gradient(180deg, rgba(200,150,74,0.16), rgba(200,150,74,0.05)) !important;
        color: var(--gold-bright) !important;
        border: 1px solid rgba(200,150,74,0.65) !important;
        border-radius: 8px !important;
        padding: 0.9rem 1.5rem !important;
        font-size: 0.98rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06),
                    0 0 26px rgba(200,150,74,0.10) !important;
        transition: all 140ms ease;
    }}
    div[class*="st-key-run_pipeline"] button p {{
        color: var(--gold-bright) !important;
        font-weight: 600;
    }}
    div[class*="st-key-run_pipeline"] button:hover {{
        background: linear-gradient(180deg, rgba(200,150,74,0.26), rgba(200,150,74,0.10)) !important;
        border-color: var(--gold-bright) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08),
                    0 0 32px rgba(200,150,74,0.22) !important;
    }}
    div[class*="st-key-run_pipeline"] button:active {{ transform: translateY(1px); }}

    /* secondary buttons (sidebar etc.) */
    .stButton > button[kind="secondary"]:not([class*="st-key-nav_"]) {{
        background: rgba(255,255,255,0.04);
        color: var(--text);
        border: 1px solid var(--gold-line);
        border-radius: 7px;
    }}

    /* ---------- instruction line ---------- */
    [data-testid="stAlert"] {{
        background: rgba(120,160,220,0.07);
        border: 1px solid rgba(130,170,230,0.16);
        border-radius: 8px;
        color: #c6d3e8;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }}
    [data-testid="stAlert"] p {{ color: #c6d3e8; font-size: 0.9rem; }}

    /* ---------- metallic framed panels ---------- */
    div[class*="st-key-output_panel"] {{
        border: 1px solid rgba(200,150,74,0.28) !important;
        border-radius: 10px !important;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.012) 100%),
            rgba(10,17,34,0.55) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05),
                    inset 0 0 40px rgba(6,10,20,0.55),
                    0 12px 30px rgba(0,0,0,0.35) !important;
        padding: 1rem 1.1rem !important;
    }}
    div[class*="st-key-output_panel"] [data-testid="stVerticalBlock"] {{
        border: none !important;
        background: transparent !important;
    }}
    .aurora-empty {{ min-height: 210px; }}

    /* hairline rule */
    hr.aurora-rule {{
        border: 0;
        height: 1px;
        margin: 0.25rem 0 1.1rem 0;
        background: linear-gradient(90deg,
            rgba(200,150,74,0) 0%,
            rgba(200,150,74,0.35) 18%,
            rgba(200,150,74,0.35) 82%,
            rgba(200,150,74,0) 100%);
    }}

    /* ---------- metrics / tables / charts ---------- */
    [data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        border: 1px solid var(--gold-line);
        border-radius: 9px;
        padding: 0.9rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] p {{
        color: var(--muted) !important;
        font-size: 0.74rem !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}
    [data-testid="stMetricValue"] {{ color: var(--gold-bright); }}
    [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {{
        border: 1px solid var(--gold-line);
        border-radius: 9px;
        overflow: hidden;
    }}
    h2, h3 {{ margin-top: 0.4rem; }}
    [data-testid="stHeadingWithActionElements"] {{ margin-bottom: 0.1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Supplementary CSS: small elements introduced by the analysis views.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stDivider"] hr { border-color: rgba(200,150,74,0.16); }
    [data-testid="stCaptionContainer"] p { color: #8b9ab5 !important; }
    [data-testid="stSpinner"] p { color: #e8c87a !important; }
    div[class*="st-key-download_"] button {
        background: linear-gradient(180deg, rgba(200,150,74,0.16), rgba(200,150,74,0.05)) !important;
        color: #e8c87a !important;
        border: 1px solid rgba(200,150,74,0.55) !important;
        border-radius: 8px !important;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    div[class*="st-key-download_"] button p { color: #e8c87a !important; }
    div[class*="st-key-download_"] button:hover { border-color: #e8c87a !important; }
    div[class*="st-key-score_headline"] button {
        background: rgba(255,255,255,0.04) !important;
        color: #e9eef7 !important;
        border: 1px solid rgba(200,150,74,0.30) !important;
        border-radius: 7px !important;
    }
    .aurora-badge-live, .aurora-badge-sample {
        display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; margin-right: 0.4rem;
    }
    .aurora-badge-live   { color: #34d399; border: 1px solid rgba(52,211,153,0.45); background: rgba(52,211,153,0.08); }
    .aurora-badge-sample { color: #fbbf24; border: 1px solid rgba(251,191,36,0.45); background: rgba(251,191,36,0.08); }

    /* ---------- editorial typography ---------- */
    h1, h2, h3, h4,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        font-family: "Crimson Pro", Georgia, "Times New Roman", serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.005em;
    }
    html, body,
    [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]) {
        font-family: "Inter", "Source Sans Pro", "Segoe UI", system-ui, sans-serif;
    }

    /* ---------- icon font guard ----------
       Puts Streamlit's icon font back *after* the two global font overrides.
       Those wildcard rules (and the `h1..h4 { font-family: ... !important }`
       editorial rule) reach every descendant, including the chrome icons, and
       would otherwise replace "Material Symbols Rounded" with Inter/Arial — at
       which point the ligature name is shown verbatim as text. !important plus
       an attribute selector outranks all of them, so an icon keeps its glyph
       even inside a heading. Covers the sidebar collapse control, expander
       chevrons, the spinner, download buttons, the dataframe toolbar, metric
       deltas and toast/dialog controls. */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-symbols-sharp,
    .material-icons,
    i.material-icons,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons", sans-serif !important;
        font-style: normal !important;
        font-weight: 400 !important;
        font-variant-ligatures: normal !important;
        font-feature-settings: "liga" 1, "clig" 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap;
    }

    /* ---------- masthead ---------- */
    .aurora-wordmark {
        font-family: "Crimson Pro", Georgia, serif;
        font-size: 1.55rem; font-weight: 600; letter-spacing: 0.24em; color: #f2f6ff;
    }
    .aurora-submark {
        font-size: 0.62rem; letter-spacing: 0.46em; color: #c8964a;
        margin-top: 0.24rem; font-weight: 600;
    }

    /* ---------- institutional section headers ---------- */
    .sec-kicker {
        color: #c8964a; font-size: 0.64rem; letter-spacing: 0.26em;
        text-transform: uppercase; font-weight: 600; margin: 0.7rem 0 0.1rem 0;
    }
    .sec-kicker::after {
        content: ""; display: inline-block; width: 2.4rem; height: 1px;
        margin-left: 0.65rem; vertical-align: middle;
        background: linear-gradient(90deg, var(--gold), transparent);
    }
    .sec-title {
        font-family: "Crimson Pro", Georgia, "Times New Roman", serif !important;
        font-size: 1.5rem !important; font-weight: 600 !important;
        color: #f2f6ff !important; letter-spacing: 0.01em !important;
        margin: 0 0 0.2rem 0 !important; line-height: 1.15 !important;
    }
    .sec-sub { color: #8b9ab5; font-size: 0.86rem; margin: 0 0 0.4rem 0; }

    /* ---------- quick-pick ticker chips ---------- */
    div[class*="st-key-qp_"] button {
        min-height: 0 !important; padding: 0.28rem 0.3rem !important;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(200,150,74,0.28) !important; border-radius: 6px !important;
    }
    div[class*="st-key-qp_"] button p { color: #8b9ab5 !important; font-size: 0.72rem !important; }
    div[class*="st-key-qp_"] button:hover { border-color: #c8964a !important; }
    div[class*="st-key-qp_"] button:hover p { color: #e8c87a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "Dashboard"
if "ticker_input" not in st.session_state:
    # Seeded here rather than as value= on the widget: the quick-pick chips
    # write this key through an on_click callback, and a widget may not carry
    # both a default value and a Session State value (Streamlit logs a policy
    # warning for every rerun if it does).
    st.session_state.ticker_input = "AAPL"
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "analysis_params" not in st.session_state:
    st.session_state.analysis_params = None
if "demo_headline" not in st.session_state:
    st.session_state.demo_headline = ""
if "demo_result" not in st.session_state:
    st.session_state.demo_result = None
if "demo_autorun" not in st.session_state:
    st.session_state.demo_autorun = False
if "corr_mode" not in st.session_state:
    st.session_state.corr_mode = "Same day"

# ----------------------------------------------------------------------------
# Header: logo (top-left)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 0.9rem 0 0.55rem 0;">
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <svg width="30" height="30" viewBox="0 0 32 32" style="flex:0 0 auto;" aria-hidden="true">
                <rect x="1.5" y="1.5" width="29" height="29" rx="6" fill="none" stroke="#c8964a" stroke-width="1.4"/>
                <rect x="8.5" y="15" width="3.4" height="8.5" fill="#8b9ab5"/>
                <rect x="14.3" y="9" width="3.4" height="14.5" fill="#c8964a"/>
                <rect x="20.1" y="12" width="3.4" height="11.5" fill="#e8c87a"/>
            </svg>
            <div style="line-height:1;">
                <div class="aurora-wordmark">AURORA</div>
                <div class="aurora-submark">FINANALYTICS</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Nav tabs (centred group of text tabs)
# ----------------------------------------------------------------------------
TABS = ["Dashboard", "Sentiment AI", "Markets", "Institutional Reports"]

with st.container(horizontal=True, horizontal_alignment="center", gap="small", key="nav_bar"):
    for label, key in zip(TABS, ("nav_dash", "nav_sent", "nav_mkt", "nav_rep")):
        if st.button(
            label,
            key=key,
            type="primary" if st.session_state.nav_tab == label else "secondary",
        ):
            st.session_state.nav_tab = label
            st.rerun()

st.markdown('<hr class="aurora-rule">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar: terminal controls
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Terminal Controls")
st.sidebar.write("")


def _pick_ticker(symbol):
    """Quick-pick handler — must be an on_click *callback*.

    Streamlit runs callbacks before the script body re-executes, so this is the
    one place where writing a widget's key is legal. Assigning
    st.session_state.ticker_input inline, in the same run that has already
    instantiated the text_input below, raises
    StreamlitWidgetAlreadyInstantiatedError — which is what happened on every
    chip click before this fix. The callback also triggers the rerun, so no
    explicit st.rerun() is needed.
    """
    st.session_state.ticker_input = symbol


ticker_symbol = st.sidebar.text_input("Asset Ticker", key="ticker_input").upper().strip()
with st.sidebar.container(horizontal=True, gap="small"):
    for _t in ("AAPL", "MSFT", "NVDA", "TSLA", "JPM"):
        st.button(_t, key=f"qp_{_t}", on_click=_pick_ticker, args=(_t,))

intraday_days = st.sidebar.slider(
    "Intraday Window (Days)", min_value=5, max_value=30, value=7,
    help="Hourly bars used by the price/sentiment overlay and the event study.",
)
daily_days = st.sidebar.slider(
    "Daily Study Window (Days)", min_value=7, max_value=60, value=30,
    help="Daily closes used for the sentiment-vs-return correlation. Widened "
         "automatically if the news feed reaches further back.",
)
st.sidebar.caption("Price bars: hourly (overlay) + daily (statistics)")
st.sidebar.caption("News: live Yahoo Finance feed + live Google News RSS")
with st.sidebar.expander("First run is slow  ·  and needs ~1 GB RAM", expanded=False):
    st.markdown(
        "FinBERT is a ~440 MB transformer that is downloaded **once** on the first "
        "scoring run and then cached. Later runs take seconds.\n\n"
        "Measured peak memory with the model resident is **~800–900 MB**, because "
        "PyTorch plus the weights sit in RAM. It runs comfortably on a 2 GB host; a "
        "1 GB free tier is tight. Nothing is sent to a third-party API — the model "
        "runs locally, and it is deliberately left unquantised because int8 "
        "quantization measurably degrades the predictions."
    )


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
def _flatten(df):
    """yfinance returns MultiIndex columns for single-ticker downloads."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _naive_utc(df):
    """Drop the exchange timezone, expressing intraday stamps in naive UTC.

    Headline timestamps from the news feeds are UTC, so intraday bars must be
    converted to UTC too — otherwise markers land in the wrong place by the
    exchange's offset (4–5 hours for US equities).
    """
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _close_frame(raw):
    """Extract a ticker-columned Close frame from a yfinance download.

    A multi-ticker download comes back with (Price, Ticker) MultiIndex columns.
    Flattening the price level *first* collapses them to ['Close', 'Close',
    'Close'] — three identically named columns holding three different indices,
    which then fails on selection and silently reports "no data" for every
    index. The Price level has to be selected before anything is flattened.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        price_level = raw.columns.get_level_values(0)
        if "Close" in set(price_level):
            return raw.xs("Close", axis=1, level=0)
        out = raw.copy()
        out.columns = price_level
        return out
    if "Close" in raw.columns:            # already-flat single-ticker frame
        return raw[["Close"]]
    return raw


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_hourly(ticker, days):
    """Hourly OHLCV bars, naive-UTC, unique and sorted (required by the
    searchsorted / get_indexer alignment used downstream)."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    try:
        raw = yf.download(
            ticker, period=f"{days}d", interval="1h", progress=False,
            auto_adjust=True, threads=False,
        )
    except Exception:
        return empty
    if raw is None or len(raw) == 0:
        return empty
    raw = _flatten(raw).dropna(how="all")
    raw = _naive_utc(raw)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_daily(ticker, days):
    """Daily adjusted closes on exchange-local dates (the definition of a
    'trading day' used for the daily return series)."""
    try:
        raw = yf.download(
            ticker, period=f"{days}d", interval="1d", progress=False,
            auto_adjust=True, threads=False,
        )
    except Exception:
        return pd.Series(dtype="float64", name="Close")
    if raw is None or len(raw) == 0:
        return pd.Series(dtype="float64", name="Close")
    raw = _flatten(raw)
    s = raw["Close"].dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "Close"
    return s


@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_name(ticker):
    """Best-effort company name, used only to sharpen the news query."""
    try:
        info = yf.Ticker(ticker).info or {}
        name = (info.get("longName") or info.get("shortName") or "").strip()
        return name if len(name) > 2 else ""
    except Exception:
        return ""


def _norm_title(title):
    """Key for cross-feed de-duplication: lowercase, punctuation stripped,
    the ' - Publisher' suffix that Google News appends removed."""
    t = (title or "").strip()
    t = re.sub(r"\s+-\s+[^-]{2,40}$", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def _news_from_yahoo(ticker, count=50):
    """Live headlines via yfinance. get_news(count=50) returns ~5x more
    history than the .news property (which caps at 10)."""
    out = []
    try:
        raw = yf.Ticker(ticker).get_news(count=count) or []
    except Exception:
        raw = []
    for item in raw:
        c = item.get("content", item) if isinstance(item, dict) else {}
        title = (c.get("title") or "").strip()
        if not title:
            continue
        ts = pd.to_datetime(c.get("pubDate"), utc=True, errors="coerce")
        if pd.isna(ts) and c.get("providerPublishTime"):
            ts = pd.to_datetime(c["providerPublishTime"], unit="s", utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        provider = c.get("provider") or {}
        url = c.get("canonicalUrl") or c.get("clickThroughUrl") or {}
        out.append(
            {
                "published": ts.tz_convert(None),
                "source": provider.get("displayName") or c.get("publisher") or "Yahoo Finance",
                "title": title,
                "summary": (c.get("summary") or c.get("description") or "").strip(),
                "link": url.get("url") if isinstance(url, dict) else "",
                "feed": "Yahoo Finance",
            }
        )
    return out


def _news_from_google_rss(query, limit=100):
    """Live headlines via Google News RSS.

    Yahoo's own feed only returns a couple of days of history, which is far too
    few days for a daily correlation. This supplementary feed reaches back
    further so the daily study has a usable n. It is labelled separately in the
    UI — it is a different source with a different relevance model.
    """
    import urllib.parse
    import urllib.request

    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
    except Exception:
        return []

    try:
        root = ET.fromstring(body)
    except Exception:
        return []

    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        if not title or not pub:
            continue
        ts = pd.to_datetime(pub, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        src = item.findtext("source") or "Google News"
        # Google News renders the publisher into the title as "Headline - Publisher".
        title = re.sub(r"\s+-\s+" + re.escape(src.strip()) + r"\s*$", "", title).strip()
        out.append(
            {
                "published": ts.tz_convert(None),
                "source": src.strip(),
                "title": title,
                "summary": (item.findtext("description") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "feed": "Google News RSS",
            }
        )
        if len(out) >= limit:
            break
    return out


def _sample_headlines(ticker):
    """Offline fallback so the demo never looks broken — clearly badged SAMPLE."""
    now = datetime.now()
    rows = [
        (now - timedelta(hours=2), "Financial Times", f"{ticker} beats quarterly revenue expectations amid high consumer demand."),
        (now - timedelta(hours=14), "Wall Street Journal", f"Supply chain disruptions raise margin concerns for {ticker} investors."),
        (now - timedelta(days=1), "Bloomberg", f"Analysts upgrade {ticker} price target citing strong structural tailwinds."),
        (now - timedelta(days=1, hours=8), "Reuters", f"Regulatory headwinds create short-term uncertainty for {ticker}."),
        (now - timedelta(days=2), "CNBC", f"{ticker} announces strategic expansion into emerging tech markets."),
        (now - timedelta(days=3), "MarketWatch", f"Market volatility impacts tech sector performance, {ticker} sees light pullback."),
        (now - timedelta(days=4), "Yahoo Finance", f"Institutional investors increase stake in {ticker} ahead of earnings."),
    ]
    return pd.DataFrame(
        [
            {"published": pd.Timestamp(t), "source": s, "title": x, "summary": "",
             "link": "", "feed": "Sample (offline)"}
            for t, s, x in rows
        ]
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker):
    """Merge every available live headline source for the ticker.

    Returns (DataFrame, provenance dict). provenance["live"] is False only when
    every network source failed and the offline sample was substituted.
    """
    rows = _news_from_yahoo(ticker, count=50)
    n_yahoo = len(rows)

    name = fetch_company_name(ticker)
    query = f'"{name}" stock when:30d' if name else f"{ticker} stock when:30d"
    rss = _news_from_google_rss(query)
    if not rss and name:
        rss = _news_from_google_rss(f"{ticker} stock when:30d")
    n_rss = len(rss)
    rows += rss

    df = pd.DataFrame(rows)
    if df.empty:
        prov = {"live": False, "feeds": ["Sample (offline)"], "n_yahoo": 0, "n_rss": 0,
                "query": query, "company": name}
        return _sample_headlines(ticker), prov

    df["_key"] = df["title"].map(_norm_title)
    # Cross-feed duplicates: prefer the wire that carries a timestamp + link.
    df["_pref"] = df["feed"].map({"Yahoo Finance": 0, "Google News RSS": 1}).fillna(2)
    df = (
        df.sort_values(["published", "_pref"])
        .drop_duplicates(subset="_key", keep="first")
        .drop(columns=["_key", "_pref"])
        .sort_values("published")
        .reset_index(drop=True)
    )
    prov = {
        "live": True,
        "feeds": [f for f, n in (("Yahoo Finance", n_yahoo), ("Google News RSS", n_rss)) if n],
        "n_yahoo": n_yahoo,
        "n_rss": n_rss,
        "query": query,
        "company": name,
    }
    return df, prov


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """FinBERT (ProsusAI/finbert): a BERT-base transformer fine-tuned for
    3-class financial sentiment. Loaded lazily and cached for the session.

    Deliberately NOT quantised. int8 dynamic quantization halves the footprint
    but measurably degrades the model — in testing it moved "Company files for
    bankruptcy amid accounting fraud investigation" from negative 0.94 to
    *neutral* 0.44, which in a finance tool is worse than being slow. Memory is
    managed with single-thread inference and low-peak loading instead.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    # One thread: FinBERT scoring is latency-tolerant and this avoids each
    # worker thread claiming its own arena on a small container.
    torch.set_num_threads(1)

    token = _hf_token()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, low_cpu_mem_usage=True, token=token
    )
    model.eval()
    return pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        top_k=None, truncation=True,
    )


def _score(analyzer, texts):
    """Full 3-class distribution for each text -> list of {label: prob} dicts."""
    raw = analyzer(list(texts))
    # A single string yields a flat list; a list of strings yields a list of lists.
    if raw and isinstance(raw[0], dict):
        raw = [raw]
    dists = []
    for item in raw:
        dists.append({d["label"].lower(): float(d["score"]) for d in item})
    return dists


def score_headlines(analyzer, news_df):
    """Add FinBERT label, confidence and a signed score to every headline.

    The signed score is P(positive) - P(negative) in [-1, +1]. Using the full
    distribution (rather than top-label confidence) keeps the magnitude
    meaningful: a genuinely mixed headline lands near zero instead of inheriting
    the full confidence of whichever class narrowly won the argmax.
    """
    news_df = news_df.copy().reset_index(drop=True)
    if news_df.empty:
        for col in ("sentiment", "confidence", "signed", "p_positive", "p_negative", "p_neutral"):
            news_df[col] = pd.Series(dtype="float64" if col != "sentiment" else "object")
        return news_df

    dists = _score(analyzer, news_df["title"].tolist())
    labels, confs, signed = [], [], []
    p_pos, p_neg, p_neu = [], [], []
    for d in dists:
        pos, neg, neu = d.get("positive", 0.0), d.get("negative", 0.0), d.get("neutral", 0.0)
        winner = max(d, key=d.get) if d else "neutral"
        labels.append(winner.capitalize())
        confs.append(round(max(d.values()) if d else 0.0, 3))
        signed.append(round(pos - neg, 4))
        p_pos.append(round(pos, 4)); p_neg.append(round(neg, 4)); p_neu.append(round(neu, 4))

    news_df["sentiment"] = labels
    news_df["confidence"] = confs
    news_df["signed"] = signed
    news_df["p_positive"] = p_pos
    news_df["p_negative"] = p_neg
    news_df["p_neutral"] = p_neu
    return news_df


def _trade_date_series(published_utc_naive):
    """Map a naive-UTC headline stamp to its exchange-local trading date.

    Bucketing news by UTC date would push after-hours US news onto the next
    calendar day — i.e. onto a trading day it did not affect.
    """
    return (
        published_utc_naive.dt.tz_localize("UTC")
        .dt.tz_convert(MARKET_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )


DAILY_COLS = ["mean_sentiment", "headlines", "return_pct", "next_return_pct"]


def build_daily(news_df, daily_close):
    """Daily mean sentiment joined to close-to-close returns.

    Returns (daily_df, n_unmatched_days). Days whose news has no trading day yet
    (e.g. pre-market news on a session that has not closed) are kept with a
    missing return and reported, never imputed.
    """
    if news_df is None or len(news_df) == 0 or daily_close is None or len(daily_close) == 0:
        return pd.DataFrame(columns=DAILY_COLS), 0

    n = news_df.copy()
    n["trade_date"] = _trade_date_series(n["published"])
    agg = n.groupby("trade_date").agg(
        mean_sentiment=("signed", "mean"), headlines=("signed", "size")
    )

    # Keep only news days that the daily price series is meant to cover.
    lo, hi = daily_close.index[0], daily_close.index[-1]
    agg = agg[(agg.index >= lo) & (agg.index <= hi + pd.Timedelta(days=1))]

    returns = (daily_close.pct_change() * 100).rename("return_pct")
    # "Next day" must mean the next *trading session*, not the next row. Rows
    # here include weekend news-days with no session, so shifting the joined
    # frame would silently turn every Friday into NaN. Shift the session-indexed
    # return series instead, so Friday -> Monday is preserved.
    next_returns = returns.shift(-1).rename("next_return_pct")
    daily = agg.join(returns, how="left").join(next_returns, how="left")
    n_unmatched = int(daily["return_pct"].isna().sum())
    return daily, n_unmatched


def _corr(x, y):
    """Pearson r with sample size and two-sided p-value.

    Returns (r, n, p) with r/p = None when there are fewer than 3 paired points
    (a correlation on 2 points is always +/-1 and means nothing).
    """
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < 3:
        return None, n, None
    a, b = pair.iloc[:, 0].to_numpy(dtype=float), pair.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None, n, None
    r = float(np.corrcoef(a, b)[0, 1])
    p = None
    if _HAVE_SCIPY:
        try:
            p = float(_scipy_stats.pearsonr(a, b)[1])
        except Exception:
            p = None
    return r, n, p


def overlay_geometry(price_df, news_df):
    """Decide where each headline belongs on the price chart — without inventing
    a timestamp.

    Headlines published after the last completed bar (all pre-market news, and
    anything after the close) cannot be drawn on the price line. The previous
    implementation clipped them onto the final bar, which stacked every marker
    at the right edge and made the event study report a fabricated 0.00%.
    """
    res = {
        "mapped": news_df.iloc[0:0],
        "unmapped": news_df.iloc[0:0],
        "pending": news_df.iloc[0:0],
        "before": news_df.iloc[0:0],
        "positions": np.array([], dtype=int),
    }
    if price_df is None or len(price_df) == 0 or news_df is None or len(news_df) == 0:
        return res

    idx = price_df.index
    pub = news_df["published"]
    in_window = (pub >= idx[0]) & (pub <= idx[-1])
    # Nearest bar within tolerance -> a marker that is actually defensible.
    nearest = idx.get_indexer(pub.to_numpy(), method="nearest",
                              tolerance=MARKER_TOLERANCE)

    mapped_mask = (in_window & pd.Series(nearest >= 0, index=news_df.index)).to_numpy()
    res["mapped"] = news_df.iloc[np.flatnonzero(mapped_mask)]
    res["positions"] = nearest[mapped_mask]

    unmapped_mask = (in_window.to_numpy()) & (~mapped_mask)
    res["unmapped"] = news_df.iloc[np.flatnonzero(unmapped_mask)]
    res["pending"] = news_df.iloc[np.flatnonzero((pub > idx[-1]).to_numpy())]
    res["before"] = news_df.iloc[np.flatnonzero((pub < idx[0]).to_numpy())]
    return res


def event_study(price_df, news_df, horizon=EVENT_HORIZON_BARS):
    """Average price drift over the `horizon` hourly bars *after* each headline.

    Entry is the first bar at or after publication (no look-ahead), and a
    headline is only used when a full forward window exists. Headlines without
    one are counted and reported instead of contributing a zero.
    """
    empty = {
        "pos": {"n": 0, "mean": None}, "neg": {"n": 0, "mean": None},
        "spread": None, "p": None, "n_used": 0, "n_skipped": 0, "detail": None,
    }
    if price_df is None or len(price_df) < horizon + 2 or news_df is None or len(news_df) == 0:
        return empty

    idx = price_df.index
    closes = price_df["Close"].to_numpy(dtype=float)
    stamps = news_df["published"].to_numpy()
    entry = idx.searchsorted(stamps, side="left")
    # A headline that predates the window would otherwise be credited with an
    # entry at bar 0 — an invented event window. Require it to be in coverage.
    valid = (
        (entry >= 0)
        & ((entry + horizon) < len(closes))
        & (stamps >= idx[0].to_datetime64())
    )
    keep = np.flatnonzero(valid)
    if len(keep) == 0:
        # Nothing measurable: say so, and say how much was dropped. Returning a
        # bare 0 here would read as "no data skipped", which is the opposite.
        empty["n_skipped"] = int(len(news_df))
        return empty

    entry_v = entry[keep]
    exit_v = entry_v + horizon
    drift = (closes[exit_v] - closes[entry_v]) / closes[entry_v] * 100.0
    signed = news_df["signed"].to_numpy(dtype=float)[keep]

    detail = news_df.iloc[keep][["published", "source", "title", "sentiment", "signed", "link"]].copy()
    detail["drift_pct"] = np.round(drift, 3)

    # A small dead zone keeps near-neutral headlines out of either bucket.
    pos_vals = drift[signed > 0.05]
    neg_vals = drift[signed < -0.05]

    out = {
        "pos": {"n": int(len(pos_vals)), "mean": float(np.mean(pos_vals)) if len(pos_vals) else None},
        "neg": {"n": int(len(neg_vals)), "mean": float(np.mean(neg_vals)) if len(neg_vals) else None},
        "spread": None, "p": None,
        "n_used": int(len(keep)),
        "n_skipped": int(len(news_df) - len(keep)),
        "detail": detail,
    }
    if len(pos_vals) and len(neg_vals):
        out["spread"] = float(np.mean(pos_vals) - np.mean(neg_vals))
        if _HAVE_SCIPY and len(pos_vals) >= 2 and len(neg_vals) >= 2:
            try:
                out["p"] = float(_scipy_stats.ttest_ind(pos_vals, neg_vals, equal_var=False)[1])
            except Exception:
                out["p"] = None
    return out


# ----------------------------------------------------------------------------
# Analysis orchestration
# ----------------------------------------------------------------------------
def run_analysis(ticker, intraday_days, daily_days):
    """Pull data, score headlines, align timestamps, compute statistics.

    Returns a plain dict (kept in session state), or None after showing an error.
    """
    price_h = fetch_price_hourly(ticker, intraday_days)
    if price_h.empty:
        error(
            f"Could not retrieve hourly price data for '{ticker}'. "
            "Check the ticker symbol (Yahoo uses suffixes such as AAPL, SAP.DE, 7203.T)."
        )
        return None

    news_df, provenance = fetch_news(ticker)
    try:
        analyzer = load_sentiment_model()
    except Exception as exc:
        error(
            "Could not load FinBERT. On a fresh machine this needs one ~440 MB "
            f"download from huggingface.co. Underlying error: {exc}"
        )
        return None

    with st.spinner("Running FinBERT over the headline tape…"):
        news_df = score_headlines(analyzer, news_df)

    # Widen the daily window if the news feed reaches further back than the
    # user's setting, so the correlation is not starved of observations.
    news_span_days = 0
    if len(news_df):
        news_span_days = int(
            (_trade_date_series(news_df["published"]).max() - _trade_date_series(news_df["published"]).min()).days
        )
    daily_days_eff = int(min(60, max(daily_days, news_span_days + 3)))
    price_d = fetch_price_daily(ticker, daily_days_eff)

    daily, n_unmatched = build_daily(news_df, price_d)

    r_same = _corr(daily["mean_sentiment"], daily["return_pct"]) if len(daily) else (None, 0, None)
    r_next = _corr(daily["mean_sentiment"], daily["next_return_pct"]) if len(daily) else (None, 0, None)
    geom = overlay_geometry(price_h, news_df)
    events = event_study(price_h, news_df)

    latest_close = float(price_h["Close"].iloc[-1])
    first_close = float(price_h["Close"].iloc[0])
    window_change = ((latest_close - first_close) / first_close * 100) if first_close else 0.0
    avg_sent = float(news_df["signed"].mean()) if len(news_df) else 0.0

    return {
        "ticker": ticker,
        "intraday_days": intraday_days,
        "daily_days": daily_days,
        "daily_days_eff": daily_days_eff,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price_h": price_h,
        "price_d": price_d,
        "news_df": news_df,
        "daily": daily,
        "daily_unmatched": n_unmatched,
        "geom": geom,
        "provenance": provenance,
        "events": events,
        "live": bool(provenance.get("live")),
        "latest_close": latest_close,
        "window_change_pct": window_change,
        "avg_sent": avg_sent,
        "sentiment_label": "Bullish" if avg_sent > 0.05 else ("Bearish" if avg_sent < -0.05 else "Neutral"),
        "n_headlines": len(news_df),
        "n_sources": int(news_df["source"].nunique()) if len(news_df) else 0,
        "latest_headline": (
            news_df["published"].max().strftime("%b %d, %H:%M UTC") if len(news_df) else "—"
        ),
        "r_same": r_same[0], "n_same": r_same[1], "p_same": r_same[2],
        "r_next": r_next[0], "n_next": r_next[1], "p_next": r_next[2],
    }


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
GRID = dict(gridcolor="rgba(139,154,181,0.10)", zerolinecolor="rgba(139,154,181,0.18)")
COL_POS, COL_NEG, COL_NEU = "#10b981", "#ef4444", "#64748b"


def _signed_colour(v):
    return COL_POS if v > 0 else (COL_NEG if v < 0 else COL_NEU)


def price_sentiment_chart(price_df, geom):
    """Price line with sentiment markers overlaid at each headline's nearest
    hourly bar, plus the signed sentiment tape underneath (shared x-axis).

    Both panels are locked to the price window, so the tape can never stretch
    the axis past the last bar while the price line stays where it is.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Price action with news-sentiment overlay", "Signed sentiment per headline"),
    )
    fig.add_trace(
        go.Scatter(x=price_df.index, y=price_df["Close"], mode="lines",
                   name="Close ($)", line=dict(color="#38bdf8", width=2)),
        row=1, col=1,
    )

    mapped = geom["mapped"]
    if len(mapped):
        pos = geom["positions"]
        mx = price_df.index[pos]
        my = price_df["Close"].to_numpy()[pos]
        hover = [
            f"<b>{t}</b><br>{s} · {p:%b %d, %H:%M} UTC<br>{lab} · signed {v:+.2f}"
            for t, s, p, lab, v in zip(
                mapped["title"], mapped["source"], mapped["published"],
                mapped["sentiment"], mapped["signed"],
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=mx, y=my, mode="markers", name="Headline (mapped)",
                marker=dict(
                    size=[8 + 10 * c for c in mapped["confidence"]],
                    color=[_signed_colour(v) for v in mapped["signed"]],
                    line=dict(width=1, color="rgba(232,200,122,0.85)"),
                ),
                hovertext=hover, hoverinfo="text",
            ),
            row=1, col=1,
        )

    # The tape shows every headline inside the price window, at its true stamp.
    on_tape = pd.concat([geom["mapped"], geom["unmapped"]]) if len(geom["unmapped"]) else geom["mapped"]
    on_tape = on_tape.sort_values("published")
    if len(on_tape):
        fig.add_trace(
            go.Bar(
                x=on_tape["published"], y=on_tape["signed"], name="Signed score",
                marker_color=[_signed_colour(v) for v in on_tape["signed"]],
                hovertext=on_tape["title"], hoverinfo="text+y",
            ),
            row=2, col=1,
        )
        if len(on_tape) >= 3:
            fig.add_trace(
                go.Scatter(
                    x=on_tape["published"],
                    y=on_tape["signed"].rolling(3, min_periods=1).mean(),
                    mode="lines", name="Rolling mean (last 3 headlines)",
                    line=dict(color="#e8c87a", width=1.5),
                ),
                row=2, col=1,
            )

    fig.update_layout(
        height=680, template="plotly_dark", showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c6d3e8"), bargap=0.25,
    )
    lo, hi = price_df.index[0], price_df.index[-1]
    fig.update_xaxes(range=[lo, hi], showgrid=False)
    fig.update_yaxes(**GRID, row=1, col=1, title_text="USD")
    fig.update_yaxes(**GRID, row=2, col=1, title_text="signed", range=[-1.05, 1.05])
    return fig


def correlation_chart(daily, mode="Same day"):
    """Scatter of daily sentiment vs daily return with an OLS fit.

    mode='Same day'  -> sentiment vs the same session's return.
    mode='Next day'  -> sentiment vs the next session's return (does tone lead price?).
    Returns None when there is nothing meaningful to draw.
    """
    y_col = "return_pct" if mode == "Same day" else "next_return_pct"
    d = daily.dropna(subset=["mean_sentiment", y_col]) if len(daily) else daily
    if len(d) < 3:
        return None

    x, y = d["mean_sentiment"].to_numpy(dtype=float), d[y_col].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=11, color="#38bdf8",
                        line=dict(width=1, color="rgba(232,200,122,0.8)")),
            text=[f"{ts:%b %d}<br>{int(h)} headline(s)" for ts, h in zip(d.index, d["headlines"])],
            hovertemplate="%{text}<br>sentiment %{x:+.2f}<br>return %{y:+.2f}%<extra></extra>",
            name="Trading day",
        )
    )

    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
    fig.add_trace(
        go.Scatter(x=xs, y=m * xs + b, mode="lines",
                   line=dict(color="#e8c87a", width=1.5, dash="dash"), name="OLS fit")
    )

    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
    label = f"r = {r:+.2f} · n = {len(d)} days" if r is not None else f"n = {len(d)} days — no variance"
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False, text=label,
        font=dict(color="#e8c87a", size=13), bgcolor="rgba(10,17,34,0.7)",
        bordercolor="rgba(200,150,74,0.4)", borderwidth=1, borderpad=6, align="left",
    )
    fig.update_layout(
        height=390, template="plotly_dark", showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c6d3e8"),
        xaxis=dict(title="Daily mean signed sentiment", **GRID),
        yaxis=dict(title=f"Daily return (%) — {mode.lower()}", **GRID),
    )
    return fig


# ----------------------------------------------------------------------------
# Small render helpers
# ----------------------------------------------------------------------------


def section_head(kicker, title, sub=None):
    """Institutional section header: a small gold kicker above a serif title."""
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="sec-kicker">{kicker}</div>'
        f'<h3 class="sec-title">{title}</h3>{sub_html}',
        unsafe_allow_html=True,
    )


_CALLOUT_STYLE = {
    "info": ("rgba(130,170,230,0.09)", "rgba(130,170,230,0.30)", "#c9d6ec", "rgba(130,170,230,0.55)"),
    "warn": ("rgba(232,200,122,0.07)", "rgba(200,150,74,0.34)", "#ecd8a8", "rgba(200,150,74,0.65)"),
    "error": ("rgba(239,110,110,0.09)", "rgba(239,110,110,0.32)", "#f0bcbc", "rgba(239,110,110,0.60)"),
}


def _callout(msg, kind="info", escape=True):
    """Icon-free notice with a thin accent bar — dependable where an icon font
    may not load, and neutral enough for an institutional product.

    `escape=True` (default) HTML-escapes the body, so user data (a ticker, an
    exception string) can never break the markup. Styled messages pass
    escape=False and supply their own inline HTML.
    """
    bg, border, fg, bar = _CALLOUT_STYLE[kind]
    body = _html.escape(msg) if escape else msg
    st.markdown(
        f'<div style="background:{bg}; border:1px solid {border}; '
        f'border-left:3px solid {bar}; border-radius:8px; '
        f'padding:0.72rem 0.95rem; color:{fg}; font-size:0.9rem; '
        f'line-height:1.6; margin-bottom:0.4rem;">{body}</div>',
        unsafe_allow_html=True,
    )


def info(msg):
    _callout(msg, "info")


def warn(msg, html=False):
    _callout(msg, "warn", escape=not html)


def error(msg):
    _callout(msg, "error")


def _hf_token():
    """HF_TOKEN from the environment or Streamlit secrets, if the user set one."""
    try:
        return os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def badge(live):
    return (
        '<span class="aurora-badge-live">LIVE</span>' if live
        else '<span class="aurora-badge-sample">SAMPLE</span>'
    )


def _fmt_r(r, n, p=None):
    if r is None:
        return f"n/a (n = {n})"
    txt = f"r = {r:+.2f} · n = {n}"
    if p is not None:
        shown = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
        txt += f" · p = {shown}" + (" (not significant)" if p > 0.05 else "")
    return txt


def _interpret(r, n, p=None):
    if r is None:
        return f"too few overlapping observations for a stable estimate (n = {n})"
    mag = "strong" if abs(r) >= 0.7 else ("moderate" if abs(r) >= 0.4 else "weak")
    sign = "positive" if r > 0.05 else ("negative" if r < -0.05 else "near-zero")
    tail = ""
    if p is not None:
        tail = (", not statistically distinguishable from zero" if p > 0.05
                else ", nominally significant at the 5% level")
    return f"a {sign}, {mag} co-movement (r = {r:+.2f}, n = {n}{tail})"


def render_provenance(res):
    """State exactly what data produced the numbers, including what was dropped."""
    prov, geom = res["provenance"], res["geom"]
    ph, pdly = res["price_h"], res["price_d"]
    news = res["news_df"]

    rows = [
        ("Hourly price bars", f"{len(ph):,} bars · {ph.index[0]:%b %d %H:%M} – {ph.index[-1]:%b %d %H:%M} UTC"
         if len(ph) else "none"),
        ("Daily price bars", f"{len(pdly):,} sessions · {pdly.index[0]:%b %d, %Y} – {pdly.index[-1]:%b %d, %Y}"
         if len(pdly) else "none"),
        ("Headline window", f"{news['published'].min():%b %d %H:%M} – {news['published'].max():%b %d %H:%M} UTC"
         if len(news) else "none"),
        ("Headline sources", ", ".join(prov.get("feeds") or ["—"])),
        ("Plotted on the price line", f"{len(geom['mapped'])} headline(s) within "
                                      f"{int(MARKER_TOLERANCE.total_seconds() // 3600)}h of an hourly bar"),
        ("In window, not on the line", f"{len(geom['unmapped'])} headline(s) published outside the trading session"),
        ("Not yet priced", f"{len(geom['pending'])} headline(s) published after the last price bar"),
        ("Event study sample", f"{res['events']['n_used']} usable · {res['events']['n_skipped']} without a full "
                               f"{EVENT_HORIZON_BARS}-bar forward window"),
        ("Daily study", f"{len(res['daily'])} news-day(s) · {len(res['daily']) - res['daily_unmatched']} matched to a "
                        f"session · {res['daily_unmatched']} not matched (weekend/holiday, or the session has not closed)"),
    ]
    with st.expander("Data coverage & provenance  (what went into — and what was left out of — the numbers)"):
        for k, v in rows:
            st.markdown(
                f"<div style='display:flex; gap:1rem; padding:0.18rem 0;'>"
                f"<div style='flex:0 0 15rem; color:#8b9ab5; font-size:0.82rem;'>{k}</div>"
                f"<div style='color:#e9eef7; font-size:0.82rem;'>{v}</div></div>",
                unsafe_allow_html=True,
            )
        if prov.get("query"):
            st.caption(f"Supplementary news query: `{prov['query']}`")
        outside = len(geom["before"]) + len(geom["pending"])
        if outside > len(geom["mapped"]):
            st.caption(
                f"{outside} of {len(news)} headlines fall outside the "
                f"{res['intraday_days']}-day intraday window. The daily study still uses all of them; "
                "widen the intraday window in the terminal controls to plot more on the price line."
            )
        if len(geom["pending"]):
            st.markdown("**Published after the last price bar — excluded from the overlay and the event study**")
            st.dataframe(
                geom["pending"][["published", "source", "title", "sentiment", "signed"]].rename(
                    columns={"published": "Published (UTC)", "source": "Source", "title": "Headline",
                             "sentiment": "Sentiment", "signed": "Signed"}
                ),
                width="stretch", hide_index=True,
                column_config={
                    "Published (UTC)": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
                    "Signed": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            st.caption(
                "These are real, already-scored headlines — they simply have no completed price bar to "
                "measure against yet. They are shown rather than silently snapped onto the last bar."
            )


def render_results(res):
    r_same, n_same, p_same = res["r_same"], res["n_same"], res["p_same"]
    ev = res["events"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Target Ticker", res["ticker"])
    with c2:
        st.metric("Latest Close", f"${res['latest_close']:,.2f}",
                  delta=f"{res['window_change_pct']:+.2f}% over window")
    with c3:
        st.metric("Aggregate NLP Sentiment", res["sentiment_label"],
                  delta=f"mean signed {res['avg_sent']:+.2f}")
    with c4:
        st.metric("Same-day Correlation",
                  f"r = {r_same:+.2f}" if r_same is not None else "n/a",
                  delta=f"{n_same} overlapping day{'s' if n_same != 1 else ''}")

    prov = res["provenance"]
    feeds = " + ".join(prov.get("feeds") or ["offline sample headlines"])
    st.markdown(
        f"{badge(res['live'])}<span style='color:#8b9ab5; font-size:0.82rem;'>"
        f"{res['n_headlines']} headlines · {res['n_sources']} sources · {feeds}"
        f" · latest {res['latest_headline']}</span>",
        unsafe_allow_html=True,
    )

    st.write("")
    section_head("Price & sentiment", "Sentiment on the price line")
    st.plotly_chart(price_sentiment_chart(res["price_h"], res["geom"]), width="stretch")
    st.caption(
        "Markers sit at each headline's nearest hourly bar (colour = FinBERT tone, size = model confidence). "
        "The tape below plots every in-window headline at its true publication stamp."
    )
    render_provenance(res)

    st.divider()
    section_head("Daily co-movement", "Does sentiment track returns?")

    daily = res["daily"]
    if len(daily) == 0:
        info(
            "No news day overlapped a completed trading session, so there is nothing to correlate. "
            "That is a real constraint of the free news feed (it returns only recent stories), not a bug — "
            "the event study below still uses every headline."
        )
    else:
        mode = st.radio(
            "Alignment", ["Same day", "Next day"], horizontal=True,
            key="corr_mode", label_visibility="collapsed",
            help="'Next day' pairs today's sentiment with the following session's return — a test of whether tone leads price.",
        )
        left, right = st.columns([3, 2])
        with left:
            fig = correlation_chart(daily, mode)
            if fig is None:
                info(
                    f"Only {len(daily)} day(s) with both sentiment and a usable return — "
                    "at least 3 are needed before a correlation line means anything."
                )
            else:
                st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("**Daily records**")
            daily_view = daily.copy()
            daily_view.index = daily_view.index.strftime("%b %d")
            st.dataframe(
                daily_view.rename(columns={
                    "mean_sentiment": "Mean sentiment", "headlines": "Headlines",
                    "return_pct": "Return %", "next_return_pct": "Next-day %",
                }),
                width="stretch", height=300,
                column_config={
                    "Mean sentiment": st.column_config.NumberColumn(format="%+.2f"),
                    "Return %": st.column_config.NumberColumn(format="%+.2f"),
                    "Next-day %": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            st.caption(
                f"Same-day: {_fmt_r(res['r_same'], res['n_same'], res['p_same'])}  \n"
                f"Next-day: {_fmt_r(res['r_next'], res['n_next'], res['p_next'])}"
            )
            st.caption(
                "Blank return cells are news days with no session (weekend/holiday) or a session that has "
                "not closed yet — left blank rather than filled in, and excluded from the correlation."
            )

        if res["n_same"] < 10 or (res["p_same"] is not None and res["p_same"] > 0.05):
            warn(
                f"<b>Read this before quoting a number.</b> The correlation rests on "
                f"n = {res['n_same']} overlapping trading day(s). With a sample this small, "
                "r is unstable and a single busy news day can move it by tenths. "
                "It is descriptive, not causal — journalists write about moves that already happened "
                "as readily as markets react to stories.",
                html=True,
            )
        st.caption("Descriptive only: over a short window, news can lag price as easily as lead it.")

    st.divider()
    section_head("Event study", "Price drift after each headline")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("After positive headlines",
                  f"{ev['pos']['mean']:+.2f}%" if ev["pos"]["mean"] is not None else "n/a",
                  delta=f"n = {ev['pos']['n']}")
    with e2:
        st.metric("After negative headlines",
                  f"{ev['neg']['mean']:+.2f}%" if ev["neg"]["mean"] is not None else "n/a",
                  delta=f"n = {ev['neg']['n']}")
    with e3:
        st.metric("Positive − negative spread",
                  f"{ev['spread']:+.2f}%" if ev["spread"] is not None else "n/a",
                  delta=f"over {EVENT_HORIZON_BARS} trading hours")
    with e4:
        st.metric("Welch p-value",
                  f"{ev['p']:.2f}" if ev["p"] is not None else "n/a",
                  delta="significant" if (ev["p"] is not None and ev["p"] <= 0.05) else "not significant")

    if ev["n_used"] == 0:
        info(
            "No headline had a full forward window of price bars, so no drift could be measured. "
            "This is the normal state for a pre-market run: today's news has not been traded yet."
        )
    else:
        st.caption(
            f"Entry is the first hourly bar at or after publication (no look-ahead); drift is measured over "
            f"{EVENT_HORIZON_BARS} bars. {ev['n_used']} headline(s) used; {ev['n_skipped']} skipped for want of a "
            "forward window. A positive spread is the direction you would expect if tone carries information — "
            "but at this sample size treat it as illustrative."
        )
        if ev["detail"] is not None and len(ev["detail"]):
            with st.expander("Per-headline drift (the raw observations behind the spread)"):
                st.dataframe(
                    ev["detail"].sort_values("published", ascending=False).rename(columns={
                        "published": "Published (UTC)", "source": "Source", "title": "Headline",
                        "sentiment": "Sentiment", "signed": "Signed", "drift_pct": f"Drift % ({EVENT_HORIZON_BARS}h)",
                        "link": "Article",
                    }),
                    width="stretch", hide_index=True,
                    column_config={
                        "Published (UTC)": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
                        "Signed": st.column_config.NumberColumn(format="%+.2f"),
                        f"Drift % ({EVENT_HORIZON_BARS}h)": st.column_config.NumberColumn(format="%+.2f"),
                        "Article": st.column_config.LinkColumn(display_text="Open"),
                    },
                )

    st.divider()
    section_head("Headline records", "Processed financial records")
    st.caption(
        "Every scored headline, including those published after the last price bar "
        "(they count toward the aggregate sentiment above, but not toward the price analyses)."
    )
    logs = res["news_df"][
        ["published", "source", "feed", "title", "sentiment", "confidence", "signed", "link"]
    ].sort_values("published", ascending=False)
    st.dataframe(
        logs.rename(columns={
            "published": "Published (UTC)", "source": "Source", "feed": "Feed", "title": "Headline",
            "sentiment": "Sentiment", "confidence": "Confidence", "signed": "Signed score", "link": "Article",
        }),
        width="stretch", hide_index=True,
        column_config={
            "Published (UTC)": st.column_config.DatetimeColumn(format="MMM D, HH:mm"),
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
            "Signed score": st.column_config.NumberColumn(format="%+.2f"),
            "Article": st.column_config.LinkColumn(display_text="Open"),
        },
    )


# ----------------------------------------------------------------------------
# Research note (Institutional Reports tab)
# ----------------------------------------------------------------------------
def build_research_note(res):
    news = res["news_df"]
    ev = res["events"]
    prov = res["provenance"]

    def bullets(df):
        if not len(df):
            return "- n/a"
        return "\n".join(
            f"- **{r.sentiment} {r.signed:+.2f}** — {r.title} *({r.source}, {r.published:%b %d %H:%M} UTC)*"
            for r in df.itertuples()
        )

    daily_rows = "\n".join(
        f"| {ts:%Y-%m-%d} | {row.mean_sentiment:+.2f} | {int(row.headlines)} | "
        f"{'—' if pd.isna(row.return_pct) else f'{row.return_pct:+.2f}%'} | "
        f"{'—' if pd.isna(row.next_return_pct) else f'{row.next_return_pct:+.2f}%'} |"
        for ts, row in res["daily"].iterrows()
    ) or "| — | — | — | — | — |"

    pos_ev = (f"**{ev['pos']['mean']:+.2f}%** (n = {ev['pos']['n']})"
              if ev["pos"]["mean"] is not None else "n/a")
    neg_ev = (f"**{ev['neg']['mean']:+.2f}%** (n = {ev['neg']['n']})"
              if ev["neg"]["mean"] is not None else "n/a")
    spread_ev = f"{ev['spread']:+.2f}%" if ev["spread"] is not None else "n/a"
    p_ev = f"{ev['p']:.3f}" if ev["p"] is not None else "n/a"
    feeds = " + ".join(prov.get("feeds") or ["offline sample headlines"])
    ph, pdly = res["price_h"], res["price_d"]

    return f"""# AURORA FINANALYTICS — Automated Research Note

**{res['ticker']}** · generated {res['generated_at']}
{'· company: ' + prov['company'] if prov.get('company') else ''}

## Data provenance

| Item | Value |
|---|---|
| Valuation feed | {'**LIVE**' if res['live'] else '**SAMPLE (offline fallback)**'} — {feeds} |
| Headlines scored | {res['n_headlines']} from {res['n_sources']} sources |
| Headline window | {news['published'].min():%Y-%m-%d %H:%M} – {news['published'].max():%Y-%m-%d %H:%M} UTC |
| Hourly price bars | {len(ph):,} bars, {ph.index[0]:%Y-%m-%d %H:%M} – {ph.index[-1]:%Y-%m-%d %H:%M} UTC |
| Daily price bars | {len(pdly):,} sessions, {pdly.index[0]:%Y-%m-%d} – {pdly.index[-1]:%Y-%m-%d} |
| Headlines on the price line | {len(res['geom']['mapped'])} mapped · {len(res['geom']['unmapped'])} unmapped |
| Headlines not yet priced | {len(res['geom']['pending'])} |
| Event-study sample | {ev['n_used']} usable · {ev['n_skipped']} skipped |

## Executive summary

- Latest close **${res['latest_close']:,.2f}** ({res['window_change_pct']:+.2f}% across the hourly window).
- Aggregate FinBERT sentiment is **{res['sentiment_label']}** (mean signed score {res['avg_sent']:+.2f} over {res['n_headlines']} headlines).
- Daily sentiment vs same-day return: {_interpret(res['r_same'], res['n_same'], res['p_same'])}.
- Daily sentiment vs next-day return: {_interpret(res['r_next'], res['n_next'], res['p_next'])}.
- Event study over {EVENT_HORIZON_BARS} trading hours: {pos_ev} after positive headlines vs {neg_ev} after negative ones; spread {spread_ev}, Welch p = {p_ev}.

> **Statistical caveat.** The daily correlation rests on n = {res['n_same']} overlapping trading day(s).
> It is reported for transparency, not as evidence of a tradeable signal. See Limitations.

## Strongest positive signals
{bullets(news.nlargest(3, 'signed') if len(news) else news)}

## Strongest negative signals
{bullets(news.nsmallest(3, 'signed') if len(news) else news)}

## Daily aggregation

| Trading day (exchange local) | Mean sentiment | Headlines | Return | Next-day return |
|---|---|---|---|---|
{daily_rows}

*Days marked "—" had no completed session — weekend/holiday news, or the current pre-market day. They are
excluded from the correlation rather than filled in.*

## Methodology

**Abstract.** This note tests whether the tone of financial news about a listed company co-moves with its
subsequent price action. It is an exploratory, small-sample study intended to demonstrate a reproducible
sentiment-to-price pipeline rather than to establish a trading signal.

**Data.**
- *Price:* hourly and daily bars from Yahoo Finance; hourly stamps converted to UTC, daily returns computed
  on exchange-local sessions with dividend/split-adjusted closes.
- *News:* live headlines from the Yahoo Finance feed plus a supplementary Google News RSS query
  (`{prov.get('query') or '—'}`). Exact duplicate titles across feeds are removed.
- *Sentiment:* FinBERT (`ProsusAI/finbert`), a BERT-base transformer fine-tuned on the Financial PhraseBank,
  run locally. Signed score = P(positive) − P(negative) ∈ [−1, +1].

**Method.**
1. Each headline is classified into positive/negative/neutral and mapped to a signed score.
2. Headlines are bucketed into exchange-local trading days and averaged.
3. Daily mean sentiment is joined to close-to-close returns and tested with Pearson correlation,
   same-day and next-day; an OLS line is drawn only for reference.
4. Overlay: each headline is drawn at the nearest hourly bar within 4 hours; headlines without a
   defensible bar are listed separately rather than moved.
5. Event study: for each headline, entry is the first hourly bar at or after publication, and drift is
   measured over {EVENT_HORIZON_BARS} bars. Positive and negative buckets are compared with Welch's t-test.

**Limitations.**
- **Small n.** The free news feeds return only recent stories; the daily correlation rests on
  n = {res['n_same']} day(s). Treat r as a description of this window, not an estimate of a stable effect.
- **Direction is ambiguous.** Coverage can follow price movements rather than cause them.
- **Model scope.** FinBERT scores general financial tone and has known failure modes on idiomatic verbs:
  it reads "smashes earnings expectations" as *negative* (P(neg) ≈ 0.93) even though "beats expectations"
  scores positive (P(pos) ≈ 0.95). Sarcasm, slang and asset-specific jargon are out of scope.
- **Feed relevance.** The supplementary feed is query-based, so unrelated-but-mentioning stories can enter.
- **No transaction costs, no tradability test, no out-of-sample validation.**

**Next steps.** Widen history with a licensed news API; restrict to earnings/M&A events; test whether any
next-day signal survives out of sample and after costs; compare across a sector basket; replace the
single-model score with an ensemble or a finance-domain LLM.

---
*Auto-generated by AURORA FINANALYTICS. Research demonstration only — not investment advice.*
"""


def build_daily_csv(res):
    d = res["daily"].copy()
    if len(d) == 0:
        return "trade_date,mean_sentiment,headlines,return_pct,next_return_pct\n"
    d.index = d.index.strftime("%Y-%m-%d")
    d = d.reset_index().rename(columns={"index": "trade_date", "trade_date": "trade_date"})
    return d.to_csv(index=False)


# ----------------------------------------------------------------------------
# Methodology copy (Sentiment AI tab)
# ----------------------------------------------------------------------------
METHODOLOGY_MD = """
### Abstract
Does the *tone* of financial news about a company line up with how its stock actually trades? Aurora
Finanalytics pulls the live headline tape for a ticker, scores every headline with a financial-sentiment
transformer (FinBERT), aligns those scores with price action — hourly for the overlay, daily for the
statistics — and tests whether daily sentiment and daily returns move together, same-day and next-day.
The question being asked is deliberately narrow and falsifiable: *is there a measurable co-movement, and
is it large enough to survive its own error bars?*

### Data
- **Price:** hourly OHLCV bars and daily adjusted closes from Yahoo Finance. Intraday stamps are converted
  to UTC so they can be aligned with headline timestamps; daily returns are computed on exchange-local
  sessions, so after-hours news is attributed to the session it actually affected.
- **News:** the live Yahoo Finance feed (`get_news`, up to 50 items) plus a supplementary live Google News
  RSS query for the company. Exact duplicate titles across the two feeds are removed. Each feed is named
  in the UI, because they are different sources with different relevance models.
- **If every network source fails,** the app falls back to a small built-in set and labels it `SAMPLE` —
  it never presents sample data as live.

### Method
1. **Scoring.** FinBERT (`ProsusAI/finbert`, a BERT-base transformer fine-tuned on the Financial PhraseBank)
   returns a full 3-class distribution per headline. The signed score is `P(positive) − P(negative)` in
   [−1, +1]. Using the whole distribution rather than the top-label confidence keeps genuinely mixed
   headlines near zero instead of letting them inherit a large magnitude from a narrow argmax win.
2. **Alignment — the part that is easy to get wrong.** A headline is drawn on the price line only if an
   hourly bar sits within 4 hours of it. Headlines published *after* the last completed bar (all pre-market
   news, and anything after the close) have no bar to sit on: they are listed explicitly under
   *Data coverage & provenance* and excluded from the price statistics. Snapping them onto the final bar —
   the obvious shortcut — stacks every marker at the right edge and makes the event study report a
   fabricated 0.00%.
3. **Aggregation.** Headlines are bucketed by **exchange-local trading date** (not UTC date, which would
   push after-hours US news onto the following day) and averaged, with headline counts retained.
4. **Testing.** Pearson correlation between daily mean sentiment and close-to-close returns — same-day and
   next-day (the "does sentiment lead price?" question) — reported with n, a p-value, and an OLS line drawn
   only for reference. Below n = 3 the app refuses to print a coefficient at all.
5. **Event study.** For every headline with a full forward window, entry is the first hourly bar at or
   after publication (no look-ahead) and drift is measured over 4 bars. Positive and negative buckets are
   compared with Welch's t-test, which does not assume equal variances.

### Limitations
- **Small samples dominate.** The free feeds return recent stories only, so the daily correlation rests on
  a handful of days. A correlation on a few points is a description of that window, not an estimate of a
  stable relationship, and the app says so next to the number.
- **Direction is ambiguous.** Journalists write about moves that already happened at least as often as
  markets react to stories, so co-movement is not evidence of predictive power.
- **Model scope.** FinBERT reads general financial tone. It has a reproducible blind spot on idiomatic
  verbs: *"smashes earnings expectations"* scores **negative** (P(neg) ≈ 0.93) while *"beats earnings
  expectations"* scores strongly **positive** (P(pos) ≈ 0.95) — same meaning, opposite sign. Try both in
  the box above. Sarcasm, slang and asset-specific jargon are out of scope.
- **Feed relevance and syndication.** The supplementary feed is query-based, so loose mentions can enter;
  only exact-title duplicates are removed, not near-duplicates.
- **No costs, no out-of-sample test, no tradability check.** Nothing here is a backtest.

### Next steps
Widen the news history with a licensed API · restrict the event study to earnings and M&A announcements ·
test whether any next-day signal survives out of sample and after transaction costs · extend the same
measurement to a sector basket to compare a sentiment factor across names.
"""

DEMO_EXAMPLES = [
    ("Beats estimates", "Nvidia beats earnings expectations as data-center demand accelerates"),
    ("Documented failure case", "Nvidia smashes earnings expectations as data-center demand accelerates"),
    ("Guidance cut", "Company cuts full-year guidance, citing weakening consumer demand"),
    ("Neutral / routine", "Company appoints new chief financial officer effective next quarter"),
]

# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------
if st.session_state.nav_tab == "Dashboard":
    with st.container(horizontal=True, horizontal_alignment="center"):
        execute = st.button("Execute Analysis Pipeline", key="run_pipeline")

    st.write("")
    info(
        "Configure your target asset in the terminal controls, then launch the engine. "
        "Live headlines are scored with FinBERT, aligned to hourly price bars, and tested against "
        "daily returns — with every sample size shown."
    )

    with st.container(border=True, key="output_panel"):
        if execute:
            try:
                res = run_analysis(ticker_symbol, intraday_days, daily_days)
                if res:
                    st.session_state.analysis = res
                    st.session_state.analysis_params = (ticker_symbol, intraday_days, daily_days)
            except Exception as exc:  # never let a demo die on a traceback
                st.session_state.analysis = None
                st.session_state.analysis_params = None
                error(f"Pipeline failed: {type(exc).__name__}: {exc}")

        res = st.session_state.analysis
        if res:
            if st.session_state.analysis_params != (ticker_symbol, intraday_days, daily_days):
                warn(
                    "Terminal controls changed since this run — the figures below reflect the previous "
                    "settings. Re-execute to refresh."
                )
            render_results(res)
        else:
            st.markdown(
                """
                <div style="padding: 1.4rem 0.4rem 1.6rem 0.4rem;">
                    <div style="color:#c8964a; font-size:0.72rem; letter-spacing:0.22em;
                                text-transform:uppercase; margin-bottom:0.7rem;">
                        Awaiting execution
                    </div>
                    <div style="color:#8b9ab5; font-size:0.88rem; line-height:1.9;">
                        <b style="color:#e9eef7;">1.</b> Fetch live headlines and hourly price bars
                        &nbsp;·&nbsp; <b style="color:#e9eef7;">2.</b> Score every headline with FinBERT
                        &nbsp;·&nbsp; <b style="color:#e9eef7;">3.</b> Align each story to the bar it landed on
                        &nbsp;·&nbsp; <b style="color:#e9eef7;">4.</b> Test daily sentiment against daily returns
                        <br>
                        <span style="font-size:0.8rem;">
                        No data leaves this app: the model runs locally, and every panel is labelled
                        LIVE or SAMPLE.
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif st.session_state.nav_tab == "Sentiment AI":
    section_head("Model", "FinBERT sentiment diagnostics")

    st.markdown(
        """
**Active model:** `ProsusAI/finbert` — a BERT-base transformer fine-tuned for 3-class financial sentiment
(positive / negative / neutral) on the Financial PhraseBank, a corpus of financial-news sentences annotated
by finance researchers. ~110M parameters, running locally on your machine — no API calls, no data leaves
the app.
"""
    )

    st.markdown("##### Try the model live")
    with st.container(horizontal=True, gap="small"):
        for i, (label, text) in enumerate(DEMO_EXAMPLES):
            if st.button(label, key=f"demo_ex_{i}"):
                st.session_state.demo_headline = text
                st.session_state.demo_autorun = True
                st.rerun()
    st.caption(
        "The failure-case chip is a documented limitation, not a typo: FinBERT reads the idiom "
        "*“smashes earnings expectations”* as negative. Being able to show the model getting a case wrong — "
        "and say why — is the point."
    )

    demo_text = st.text_input(
        "Paste any financial headline and FinBERT will score it:",
        placeholder="e.g. Nvidia beats earnings expectations as data-center demand accelerates",
        label_visibility="collapsed",
        key="demo_headline",
    )

    autorun = bool(st.session_state.pop("demo_autorun", False))
    clicked = st.button("Score this headline", key="score_headline")
    if clicked or autorun:
        if not demo_text.strip():
            warn("Type a headline first.")
        else:
            try:
                with st.spinner("Scoring…"):
                    dist = _score(load_sentiment_model(), [demo_text.strip()])[0]
                st.session_state.demo_result = {"text": demo_text.strip(), "dist": dist}
            except Exception as exc:
                st.session_state.demo_result = None
                error(f"Could not score that headline: {type(exc).__name__}: {exc}")

    demo_result = st.session_state.demo_result
    if demo_result and demo_result["text"] == demo_text.strip():
        dist = demo_result["dist"]
        pos, neg, neu = dist.get("positive", 0.0), dist.get("negative", 0.0), dist.get("neutral", 0.0)
        signed = pos - neg
        winner = max(dist, key=dist.get) if dist else "neutral"
        d1, d2, d3 = st.columns(3)
        d1.metric("Classification", winner.capitalize())
        d2.metric("Model confidence", f"{max(dist.values()):.1%}" if dist else "n/a")
        d3.metric("Signed score", f"{signed:+.2f}")
        st.markdown(
            f"<div style='margin-top:0.5rem; font-size:0.82rem; color:#8b9ab5;'>"
            f"P(positive) {pos:.3f} &nbsp;·&nbsp; P(negative) {neg:.3f} &nbsp;·&nbsp; P(neutral) {neu:.3f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(pos, 0.0), 1.0), text=f"P(positive) {pos:.1%}")
        st.progress(min(max(neg, 0.0), 1.0), text=f"P(negative) {neg:.1%}")
        st.progress(min(max(neu, 0.0), 1.0), text=f"P(neutral) {neu:.1%}")
        st.caption(
            "The signed score is P(positive) − P(negative), so a headline the model finds genuinely "
            "ambiguous lands near zero rather than inheriting a large magnitude from a narrow argmax win."
        )

    st.divider()
    section_head("Methodology", "Research methodology")
    with st.container(border=True):
        st.markdown(METHODOLOGY_MD)

elif st.session_state.nav_tab == "Markets":
    section_head("Markets", "Macro market overview")
    indices = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}

    @st.cache_data(ttl=900, show_spinner=False)
    def _fetch_indices():
        raw = yf.download(list(indices.values()), period="5d", interval="1d",
                          progress=False, auto_adjust=True, threads=False)
        return _naive_utc(_close_frame(raw))

    try:
        market_data = _fetch_indices()
        if market_data is None or market_data.empty:
            raise ValueError("empty response from Yahoo Finance")

        cols = st.columns(3)
        for col, (name, sym) in zip(cols, indices.items()):
            if sym not in market_data.columns:
                col.metric(name, "n/a", delta="no data")
                continue
            series = market_data[sym].dropna()
            if series.empty:
                col.metric(name, "n/a", delta="no data")
                continue
            delta = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
            col.metric(name, f"{series.iloc[-1]:,.0f}", delta=f"{delta:+.2f}% (5 sessions)")

        st.markdown(
            f"{badge(True)}<span style='color:#8b9ab5; font-size:0.82rem;'>"
            f"Daily closes via Yahoo Finance · {market_data.dropna(how='all').index[-1]:%b %d, %Y}</span>",
            unsafe_allow_html=True,
        )

        norm = market_data / market_data.bfill().iloc[0] * 100
        fig = go.Figure()
        for name, sym in indices.items():
            if sym in norm.columns:
                fig.add_trace(go.Scatter(x=norm.index, y=norm[sym], mode="lines",
                                         name=name, line=dict(width=2)))
        fig.update_layout(
            height=390, template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c6d3e8"),
            yaxis=dict(title="Indexed to 100 at window start", **GRID),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("All three indices are rebased to 100 at the start of the window, so the lines compare "
                   "percentage moves rather than index levels.")
        st.dataframe(
            market_data.tail().rename(columns={v: k for k, v in indices.items()}),
            width="stretch",
            column_config={n: st.column_config.NumberColumn(n, format="%,.2f") for n in indices},
        )
    except Exception as exc:
        error(f"Could not load index data: {type(exc).__name__}: {exc}")
        st.caption("This tab only depends on Yahoo Finance. Retry in a moment — the endpoint rate-limits "
                   "bursty requests.")

elif st.session_state.nav_tab == "Institutional Reports":
    section_head("Research note", "Automated research note")
    res = st.session_state.analysis
    if not res:
        info(
            "Run the analysis pipeline on the Dashboard first — this note is generated directly from its "
            "output, so it can only be as current as the last run."
        )
    else:
        note = build_research_note(res)
        c1, c2 = st.columns([2, 2])
        with c1:
            st.download_button(
                "Download research note (.md)",
                data=note,
                file_name=f"AURORA_{res['ticker']}_{datetime.now():%Y%m%d}.md",
                mime="text/markdown",
                key="download_note",
            )
        with c2:
            st.download_button(
                "Download daily series (.csv)",
                data=build_daily_csv(res),
                file_name=f"AURORA_{res['ticker']}_daily_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                key="download_csv",
            )
        # st.caption does not render HTML, so the badge must go through markdown.
        st.markdown(
            f"<span style='color:#8b9ab5; font-size:0.82rem;'>Auto-generated from the "
            f"{res['generated_at']} run —</span> {badge(res['live'])}"
            f"<span style='color:#8b9ab5; font-size:0.82rem;'>"
            f"{'live feeds' if res['live'] else 'offline sample data'}.</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        with st.container(border=True):
            st.markdown(note)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown('<hr class="aurora-rule" style="margin-top:2rem;">', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#6b7a94; font-size:0.74rem; padding-bottom:1.4rem;'>"
    "AURORA FINANALYTICS · research demonstration · not investment advice · "
    "sentiment scored locally with FinBERT · every figure carries its sample size"
    "</div>",
    unsafe_allow_html=True,
)
