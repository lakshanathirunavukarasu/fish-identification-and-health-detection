import streamlit as st
import streamlit.components.v1 as components
from api_client import predict_image
from PIL import Image
import time
import requests
import pandas as pd
import numpy as np

# ── Volume Estimation Helper ──────────────────────────────────────────────────
def estimate_volume(image: Image.Image) -> dict:
    import numpy as np
    from PIL import ImageFilter

    img_gray = image.convert("L")
    img_arr  = np.array(img_gray)

    threshold = np.mean(img_arr) * 0.85
    mask = img_arr < threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        row_min, row_max = 0, img_arr.shape[0]
        col_min, col_max = 0, img_arr.shape[1]
    else:
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

    pixel_length = col_max - col_min
    pixel_width  = row_max - row_min

    W, H = image.size

    ASSUMED_REAL_LENGTH_CM = 25.0
    scale_cm_per_px = ASSUMED_REAL_LENGTH_CM / (W * 0.65)

    length_cm = round(pixel_length * scale_cm_per_px, 1)
    width_cm  = round(pixel_width  * scale_cm_per_px, 1)
    depth_cm  = round(width_cm * 0.6, 1)

    a = length_cm / 2
    b = width_cm  / 2
    volume_cm3    = round((4/3) * 3.14159 * a * b * b, 1)
    volume_liters = round(volume_cm3 / 1000, 3)
    weight_est_g  = round(volume_cm3 * 1.05, 1)
    weight_est_kg = round(weight_est_g / 1000, 3)

    return {
        "length_cm":     length_cm,
        "width_cm":      width_cm,
        "depth_cm":      depth_cm,
        "volume_cm3":    volume_cm3,
        "volume_liters": volume_liters,
        "weight_est_g":  weight_est_g,
        "weight_est_kg": weight_est_kg,
    }

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="NEURAL NET BASED FISH IDENTIFICATION",
    page_icon="🐡",
    layout="wide",
)

# ================= CUSTOM STYLE =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --pearl:        #fafcff;
        --white:        #ffffff;
        --ocean-50:     #eff8ff;
        --ocean-100:    #dbeeff;
        --ocean-300:    #7ec8e3;
        --ocean-500:    #1a8db5;
        --ocean-700:    #0e5f7a;
        --teal-400:     #2dd4bf;
        --teal-600:     #0d9488;
        --coral-400:    #fb7185;
        --amber-400:    #fbbf24;
        --ink:          #0f2a3a;
        --ink-60:       #3d6070;
        --ink-30:       #8aaabb;
        --border-soft:  rgba(30,120,160,0.12);
        --shadow-xs:    0 1px 4px rgba(14,95,122,0.06);
        --shadow-sm:    0 4px 16px rgba(14,95,122,0.10);
        --shadow-md:    0 8px 32px rgba(14,95,122,0.14);
        --shadow-lg:    0 16px 56px rgba(14,95,122,0.18);
        --r-card:       20px;
        --r-pill:       999px;
    }

    html, body, .stApp {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--ink) !important;
    }

    /* ── REMOVE TOP WHITE SPACE ── */
    .stApp > header { display: none !important; }
    #root > div:first-child { padding-top: 0 !important; }
    .main .block-container {
        padding-top: 0.8rem !important;
        padding-left: 3.5rem;
        padding-right: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        position: relative;
        z-index: 1;
    }
    /* Remove Streamlit default top padding injected via inline style */
    section.main > div { padding-top: 0 !important; }

    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 0% 0%,   rgba(30,141,181,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 100% 0%,  rgba(45,212,191,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 70% 60% at 50% 100%, rgba(251,113,133,0.05) 0%, transparent 60%),
            linear-gradient(160deg, #f5faff 0%, #ffffff 45%, #f0fbf9 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }

    .main .block-container::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(circle at 1px 1px, rgba(30,141,181,0.055) 1px, transparent 0);
        background-size: 28px 28px;
        pointer-events: none;
        z-index: -1;
    }

    h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        line-height: 1.15 !important;
        background: linear-gradient(130deg, var(--ocean-700) 0%, var(--ocean-500) 40%, var(--teal-400) 80%);
        background-size: 200%;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: titleWave 6s ease-in-out infinite alternate;
        letter-spacing: -0.3px;
    }

    h2 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        color: var(--ocean-700) !important;
        font-size: 1.35rem !important;
    }

    h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--ink-60) !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    @keyframes titleWave {
        from { background-position: 0% center; }
        to   { background-position: 100% center; }
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0b3547 0%, #0e4f65 40%, #0d6e8a 75%, #1a8db5 100%) !important;
        border-right: none !important;
        box-shadow: 6px 0 40px rgba(11,53,71,0.25) !important;
        position: relative;
        overflow: hidden;
    }

    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        inset: 0;
        background-image:
            radial-gradient(circle 5px at 20% 25%, rgba(255,255,255,0.12) 0%, transparent 100%),
            radial-gradient(circle 3px at 70% 15%, rgba(255,255,255,0.09) 0%, transparent 100%),
            radial-gradient(circle 4px at 85% 55%, rgba(255,255,255,0.08) 0%, transparent 100%),
            radial-gradient(circle 6px at 35% 75%, rgba(255,255,255,0.07) 0%, transparent 100%),
            radial-gradient(circle 3px at 55% 90%, rgba(255,255,255,0.10) 0%, transparent 100%);
        animation: sidebarBubbles 8s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stSidebar"]::after {
        content: '';
        position: absolute;
        top: 0; left: -30%; right: -30%;
        height: 220px;
        background: radial-gradient(ellipse at 50% -10%, rgba(125,220,255,0.18) 0%, transparent 70%);
        animation: surfaceLight 5s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes sidebarBubbles {
        0%   { transform: translateY(0);    opacity: 0.8; }
        100% { transform: translateY(-18px); opacity: 1;   }
    }
    @keyframes surfaceLight {
        0%   { opacity: 0.6; transform: scaleX(1);   }
        100% { opacity: 1;   transform: scaleX(1.1); }
    }

    [data-testid="stSidebar"] > div { position: relative; z-index: 1; }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0f4ff !important;
        -webkit-text-fill-color: #e0f4ff !important;
        background: none !important;
        font-family: 'Playfair Display', serif !important;
        animation: none !important;
        text-shadow: 0 2px 12px rgba(0,0,0,0.25);
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stText {
        color: rgba(200,235,255,0.85) !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stSidebar"] hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent) !important;
        opacity: 1 !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--white) !important;
        border-radius: var(--r-card) !important;
        padding: 6px !important;
        gap: 4px !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--border-soft) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        color: var(--ink-60) !important;
        background: transparent !important;
        border-radius: 14px !important;
        padding: 9px 22px !important;
        border: none !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.1px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--ocean-50) !important;
        color: var(--ocean-700) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--ocean-500), var(--ocean-700)) !important;
        color: white !important;
        box-shadow: 0 4px 18px rgba(14,95,122,0.30) !important;
        font-weight: 600 !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(239,248,255,0.9), rgba(240,251,249,0.9)) !important;
        border: 2px dashed rgba(30,141,181,0.3) !important;
        border-radius: var(--r-card) !important;
        padding: 32px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-xs);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--ocean-500) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-2px) !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        background: linear-gradient(135deg, var(--ocean-500) 0%, var(--ocean-700) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 28px !important;
        box-shadow: 0 4px 18px rgba(14,95,122,0.25) !important;
        transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        letter-spacing: 0.2px;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 32px rgba(14,95,122,0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(0.99) !important;
    }

    /* ── PROGRESS BARS ── */
    [data-testid="stProgress"] > div > div {
        background: var(--ocean-100) !important;
        border-radius: var(--r-pill) !important;
        height: 9px !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, var(--ocean-500), var(--teal-400)) !important;
        border-radius: var(--r-pill) !important;
        box-shadow: 0 2px 8px rgba(30,141,181,0.30) !important;
        animation: shimmerBar 2s linear infinite;
        background-size: 200% 100%;
    }

    @keyframes shimmerBar {
        0%   { background-position: 200% center; }
        100% { background-position: 0% center; }
    }

    /* ── ALERTS ── */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border-left-width: 5px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        box-shadow: var(--shadow-xs) !important;
        animation: popIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }

    @keyframes popIn {
        from { opacity: 0; transform: scale(0.94) translateY(8px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {
        border-radius: var(--r-card) !important;
        overflow: hidden !important;
        border: 1px solid var(--border-soft) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── COLUMN CARDS ── */
    [data-testid="column"] {
        background: var(--white);
        border-radius: var(--r-card);
        padding: 20px;
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-sm);
        animation: riseIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(22px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── REPORT BOX ── */
    .report-box {
        background: var(--white) !important;
        padding: 28px !important;
        border-radius: var(--r-card) !important;
        border: 1px solid var(--border-soft) !important;
        box-shadow: var(--shadow-md) !important;
        animation: riseIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }

    /* ── DIVIDERS ── */
    hr {
        border: none !important;
        height: 1.5px !important;
        background: linear-gradient(90deg, transparent 0%, var(--ocean-300) 30%, var(--teal-400) 70%, transparent 100%) !important;
        opacity: 0.35 !important;
        margin: 1.6rem 0 !important;
        border-radius: 2px !important;
    }

    /* ── IMAGE ── */
    [data-testid="stImage"] img {
        border-radius: 16px !important;
        border: 2px solid var(--border-soft) !important;
        box-shadow: 0 10px 36px rgba(14,95,122,0.14) !important;
        transition: transform 0.35s ease, box-shadow 0.35s ease !important;
    }

    [data-testid="stImage"] img:hover {
        transform: scale(1.015) !important;
        box-shadow: 0 16px 48px rgba(14,95,122,0.22) !important;
    }

    /* ── TEXT ── */
    .stMarkdown p, .stText, label {
        color: var(--ink-60) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.92rem !important;
    }

    strong { color: var(--ink) !important; font-weight: 700 !important; }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--ink-30) !important;
        font-size: 0.78rem !important;
    }

    /* ── RIGHT COLUMN COMPACT ── */
    [data-testid="column"]:nth-child(2) { padding: 14px 16px !important; }
    [data-testid="column"]:nth-child(2) h3 { font-size: 0.78rem !important; margin-bottom: 4px !important; }
    [data-testid="column"]:nth-child(2) .stMarkdown p { font-size: 0.82rem !important; margin-bottom: 2px !important; }
    [data-testid="column"]:nth-child(2) [data-testid="stProgress"] > div > div { height: 7px !important; }
    [data-testid="column"]:nth-child(2) [data-testid="stCaptionContainer"] { font-size: 0.72rem !important; margin-top: 1px !important; }
    [data-testid="column"]:nth-child(2) [data-testid="stAlert"] { padding: 8px 12px !important; font-size: 0.82rem !important; }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--ocean-50); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--ocean-300), var(--teal-400));
        border-radius: 3px;
    }

    /* ── SIDEBAR ANIMATED ELEMENTS ── */
    @keyframes pulseRing  { 0%,100%{transform:scale(1);opacity:.6} 50%{transform:scale(1.08);opacity:1} }
    @keyframes scanLine   { 0%{top:0%;opacity:.7} 100%{top:100%;opacity:0} }
    @keyframes blinkDot   { 0%,100%{opacity:1} 50%{opacity:.2} }
    @keyframes barFill    { from{width:0%} to{width:var(--fill)} }
    @keyframes barFill1   { from{width:0} to{width:92%} }
    @keyframes barFill2   { from{width:0} to{width:87%} }
    @keyframes barFill3   { from{width:0} to{width:78%} }
    @keyframes barFill4   { from{width:0} to{width:95%} }
    @keyframes fadeUp     { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
    @keyframes countUp    { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }
    @keyframes rotateSpin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
    @keyframes waveFlow   { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════
st.sidebar.markdown("""
<div style="text-align:center;padding:20px 10px 8px;position:relative;">
  <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;
              color:#e0f4ff;letter-spacing:0.5px;text-shadow:0 2px 10px rgba(0,0,0,0.35);
              margin-bottom:2px;line-height:1.4;">
    🐡 NEURAL NETWORK BASED<br>REAL TIME FISH <br>IDENTIFICATION, HEALTH<br>&amp; VOLUME ESTIMATION
  </div>
  <div style="font-size:0.62rem;color:rgba(180,225,255,0.60);letter-spacing:2.5px;
              text-transform:uppercase;margin-top:6px;">
    Fish Identification AI
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

import base64 as _b64
_svg = """<svg viewBox="0 0 220 420" xmlns="http://www.w3.org/2000/svg" width="220" height="420">
  <text x="110" y="16" text-anchor="middle" font-size="10" fill="rgba(180,225,255,0.65)" font-family="sans-serif" letter-spacing="1.5" font-weight="600">&#x25C8; NEURAL SCAN ENGINE</text>
  <circle cx="110" cy="80" r="50" fill="none" stroke="rgba(45,212,191,0.18)" stroke-width="1.5"><animate attributeName="r" values="50;55;50" dur="3s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.5;1;0.5" dur="3s" repeatCount="indefinite"/></circle>
  <circle cx="110" cy="80" r="37" fill="none" stroke="rgba(45,212,191,0.30)" stroke-width="1.5"><animate attributeName="r" values="37;41;37" dur="3s" begin="0.6s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.5;1;0.5" dur="3s" begin="0.6s" repeatCount="indefinite"/></circle>
  <circle cx="110" cy="80" r="24" fill="none" stroke="rgba(45,212,191,0.46)" stroke-width="1.5"><animate attributeName="r" values="24;27;24" dur="3s" begin="1.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.5;1;0.5" dur="3s" begin="1.2s" repeatCount="indefinite"/></circle>
  <line x1="110" y1="32" x2="110" y2="128" stroke="rgba(45,212,191,0.14)" stroke-width="0.8"/>
  <line x1="62" y1="80" x2="158" y2="80" stroke="rgba(45,212,191,0.14)" stroke-width="0.8"/>
  <line x1="110" y1="80" x2="110" y2="32" stroke="rgba(45,212,191,0.85)" stroke-width="1.8" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 110 80" to="360 110 80" dur="3s" repeatCount="indefinite"/></line>
  <line x1="110" y1="80" x2="128" y2="36" stroke="rgba(45,212,191,0.22)" stroke-width="1.2" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 110 80" to="360 110 80" dur="3s" repeatCount="indefinite"/></line>
  <circle cx="110" cy="80" r="5" fill="#2dd4bf"><animate attributeName="opacity" values="1;0.2;1" dur="1.5s" repeatCount="indefinite"/></circle>
  <circle cx="130" cy="58" r="4" fill="#7ee8fa"><animate attributeName="opacity" values="1;0.1;1" dur="2.1s" repeatCount="indefinite"/></circle>
  <circle cx="88" cy="98" r="3.5" fill="#fb7185"><animate attributeName="opacity" values="1;0.1;1" dur="1.8s" begin="0.9s" repeatCount="indefinite"/></circle>
  <circle cx="84" cy="64" r="3.5" fill="#fbbf24"><animate attributeName="opacity" values="1;0.1;1" dur="2.5s" begin="1.3s" repeatCount="indefinite"/></circle>
  <circle cx="96" cy="144" r="4" fill="#2dd4bf"><animate attributeName="opacity" values="1;0.15;1" dur="1.6s" repeatCount="indefinite"/></circle>
  <text x="104" y="149" font-size="10" fill="#2dd4bf" font-family="sans-serif" letter-spacing="1.5" font-weight="600">SYSTEM ACTIVE</text>
  <line x1="10" y1="162" x2="210" y2="162" stroke="rgba(255,255,255,0.12)" stroke-width="0.9"/>
  <text x="10" y="178" font-size="10.5" fill="rgba(180,225,255,0.70)" font-family="sans-serif" letter-spacing="1.5" font-weight="700">&#x25C8; MODEL PERFORMANCE</text>
  <text x="10" y="197" font-size="11" fill="rgba(220,240,255,0.90)" font-family="sans-serif" font-weight="500">Species Accuracy</text>
  <text x="210" y="197" font-size="11" fill="#7ee8fa" font-family="sans-serif" text-anchor="end" font-weight="800">92%</text>
  <rect x="10" y="201" width="200" height="7" rx="3.5" fill="rgba(255,255,255,0.09)"/>
  <rect x="10" y="201" width="0" height="7" rx="3.5" fill="url(#g1)"><animate attributeName="width" from="0" to="184" dur="1.5s" begin="0.3s" fill="freeze"/></rect>
  <text x="10" y="223" font-size="11" fill="rgba(220,240,255,0.90)" font-family="sans-serif" font-weight="500">Health Detection</text>
  <text x="210" y="223" font-size="11" fill="#7ee8fa" font-family="sans-serif" text-anchor="end" font-weight="800">87%</text>
  <rect x="10" y="227" width="200" height="7" rx="3.5" fill="rgba(255,255,255,0.09)"/>
  <rect x="10" y="227" width="0" height="7" rx="3.5" fill="url(#g2)"><animate attributeName="width" from="0" to="174" dur="1.5s" begin="0.5s" fill="freeze"/></rect>
  <text x="10" y="249" font-size="11" fill="rgba(220,240,255,0.90)" font-family="sans-serif" font-weight="500">Volume Estimation</text>
  <text x="210" y="249" font-size="11" fill="#7ee8fa" font-family="sans-serif" text-anchor="end" font-weight="800">78%</text>
  <rect x="10" y="253" width="200" height="7" rx="3.5" fill="rgba(255,255,255,0.09)"/>
  <rect x="10" y="253" width="0" height="7" rx="3.5" fill="url(#g3)"><animate attributeName="width" from="0" to="156" dur="1.5s" begin="0.7s" fill="freeze"/></rect>
  <text x="10" y="275" font-size="11" fill="rgba(220,240,255,0.90)" font-family="sans-serif" font-weight="500">Confidence Score</text>
  <text x="210" y="275" font-size="11" fill="#7ee8fa" font-family="sans-serif" text-anchor="end" font-weight="800">95%</text>
  <rect x="10" y="279" width="200" height="7" rx="3.5" fill="rgba(255,255,255,0.09)"/>
  <rect x="10" y="279" width="0" height="7" rx="3.5" fill="url(#g4)"><animate attributeName="width" from="0" to="190" dur="1.5s" begin="0.9s" fill="freeze"/></rect>
  <line x1="10" y1="296" x2="210" y2="296" stroke="rgba(255,255,255,0.12)" stroke-width="0.9"/>
  <text x="10" y="311" font-size="10.5" fill="rgba(180,225,255,0.70)" font-family="sans-serif" letter-spacing="1.5" font-weight="700">&#x25C8; TECH STACK</text>
  <rect x="10" y="318" width="58" height="17" rx="8" fill="rgba(45,212,191,0.15)" stroke="rgba(45,212,191,0.40)" stroke-width="0.9"/><text x="39" y="330" text-anchor="middle" font-size="9.5" fill="#7ee8fa" font-family="sans-serif" font-weight="600">TensorFlow</text>
  <rect x="74" y="318" width="36" height="17" rx="8" fill="rgba(99,102,241,0.15)" stroke="rgba(99,102,241,0.40)" stroke-width="0.9"/><text x="92" y="330" text-anchor="middle" font-size="9.5" fill="#a5b4fc" font-family="sans-serif" font-weight="600">Keras</text>
  <rect x="116" y="318" width="40" height="17" rx="8" fill="rgba(251,146,60,0.15)" stroke="rgba(251,146,60,0.40)" stroke-width="0.9"/><text x="136" y="330" text-anchor="middle" font-size="9.5" fill="#fdba74" font-family="sans-serif" font-weight="600">FastAPI</text>
  <rect x="162" y="318" width="50" height="17" rx="8" fill="rgba(248,113,130,0.15)" stroke="rgba(248,113,130,0.40)" stroke-width="0.9"/><text x="187" y="330" text-anchor="middle" font-size="9.5" fill="#fda4af" font-family="sans-serif" font-weight="600">Streamlit</text>
  <rect x="10" y="341" width="28" height="17" rx="8" fill="rgba(250,204,21,0.15)" stroke="rgba(250,204,21,0.40)" stroke-width="0.9"/><text x="24" y="353" text-anchor="middle" font-size="9.5" fill="#fde68a" font-family="sans-serif" font-weight="600">PIL</text>
  <rect x="44" y="341" width="44" height="17" rx="8" fill="rgba(30,141,181,0.15)" stroke="rgba(30,141,181,0.45)" stroke-width="0.9"/><text x="66" y="353" text-anchor="middle" font-size="9.5" fill="#7ec8e3" font-family="sans-serif" font-weight="600">Python</text>
  <defs>
    <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#1a8db5"/><stop offset="100%" stop-color="#2dd4bf"/></linearGradient>
    <linearGradient id="g2" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#0d9488"/><stop offset="100%" stop-color="#2dd4bf"/></linearGradient>
    <linearGradient id="g3" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#a78bfa"/></linearGradient>
    <linearGradient id="g4" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#fbbf24"/></linearGradient>
  </defs>
</svg>"""
_b64_svg = _b64.b64encode(_svg.encode("utf-8")).decode("utf-8")
_data_uri = f"data:image/svg+xml;base64,{_b64_svg}"
st.sidebar.markdown(f'<img src="{_data_uri}" width="100%" style="display:block;margin:0 auto;max-width:220px;">', unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.22);
            border-left:4px solid #2dd4bf;border-radius:12px;padding:12px 14px;margin:4px 0;">
  <span style="color:#ffffff;font-family:'DM Sans',sans-serif;font-size:0.85rem;font-weight:500;">
    📡 Upload a fish image to begin species &amp; health detection.
  </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center;padding:12px 0 6px;">
  <div style="font-size:0.67rem;color:rgba(150,210,240,0.45);letter-spacing:1px;">
    © 2026 · Bharathiar University
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  MAIN AREA HEADER BANNER
# ══════════════════════════════════════════
st.markdown("""
<div style="
  background:linear-gradient(135deg, #0b3547 0%, #0e5f7a 50%, #1a8db5 100%);
  border-radius:24px;
  padding:28px 36px;
  margin-bottom:28px;
  position:relative;
  overflow:hidden;
  box-shadow:0 10px 40px rgba(14,95,122,0.22);
">
  <svg style="position:absolute;right:24px;top:50%;transform:translateY(-50%);opacity:0.15;pointer-events:none;"
       width="280" height="100" viewBox="0 0 280 100" xmlns="http://www.w3.org/2000/svg">
    <polygon points="28,50 8,36 8,64" fill="white"/>
    <ellipse cx="58" cy="50" rx="34" ry="17" fill="white"/>
    <circle cx="86" cy="45" r="4.5" fill="rgba(0,0,0,0.35)"/>
    <path d="M44,42 Q52,35 60,42" fill="rgba(255,255,255,0.3)" stroke="none"/>
    <polygon points="162,22 148,15 148,29" fill="white"/>
    <ellipse cx="180" cy="22" rx="22" ry="10" fill="white"/>
    <circle cx="199" cy="20" r="3" fill="rgba(0,0,0,0.35)"/>
    <polygon points="210,74 200,68 200,80" fill="white"/>
    <ellipse cx="223" cy="74" rx="16" ry="8" fill="white"/>
    <circle cx="237" cy="72" r="2.2" fill="rgba(0,0,0,0.35)"/>
    <circle cx="120" cy="18" r="4.5" fill="none" stroke="white" stroke-width="1.5" opacity="0.55"/>
    <circle cx="138" cy="32" r="2.8" fill="none" stroke="white" stroke-width="1.2" opacity="0.45"/>
    <circle cx="250" cy="38" r="3.2" fill="none" stroke="white" stroke-width="1.2" opacity="0.40"/>
    <circle cx="265" cy="20" r="2"   fill="none" stroke="white" stroke-width="1"   opacity="0.35"/>
  </svg>
  <div style="position:relative;z-index:1;">
    <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:rgba(160,230,255,0.80);
                letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;">
      Neural Network · Transfer Learning · Real-Time
    </div>
    <div style="font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:800;
                color:white;line-height:1.2;text-shadow:0 2px 14px rgba(0,0,0,0.28);">
      Fish Identification AI
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:0.9rem;
                color:rgba(190,235,255,0.82);margin-top:7px;">
      Identify species &amp; diagnose health conditions from a single image
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🪟 Home",
    "🔬 Species",
    "🩺 Health",
    "🗂️ Batch Analysis",
    "📊 About Project"
])

# =====================================================
# ===================== HOME TAB ======================
# =====================================================
with tab1:

    st.title("🐡 Fish ANALYSIS")
    st.subheader("Smart Fish Species & Health, Volume Detection System")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📡 Upload Fish Image",
        type=["jpg", "jpeg", "png", "jfif"]
    )

    if uploaded_file:

        col1, col2 = st.columns([1.6, 1])

        image = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)

        with col2:
            with st.spinner("🧬 Analyzing Image..."):
                time.sleep(1)
                result = predict_image(uploaded_file)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Prediction Completed 🎯")
                st.markdown("### 🧾 Diagnosis Report")

                st.write(f"🔬 **Species:** {result['species']}")
                if result.get('scientific_name'):
                    st.caption(f"🔖 Scientific Name: *{result['scientific_name']}*")
                st.progress(int(result['species_confidence']))
                st.caption(f"Confidence: {result['species_confidence']:.2f}%")

                st.markdown("---")

                st.write(f"🩺 **Health Condition:** {result['health']}")
                if result.get('health_details'):
                    st.caption(f"📂 Category: {result['health_details'].get('category','—')}  |  🩹 {result['health_details'].get('symptoms','—')}")
                st.progress(int(result['health_confidence']))
                st.caption(f"Confidence: {result['health_confidence']:.2f}%")

                st.markdown("---")

                vol = estimate_volume(image)
                st.markdown("### ⚖️ Volume Estimation")
                st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(14,95,122,0.07),rgba(45,212,191,0.05));
            border:1px solid rgba(30,120,160,0.15);border-radius:14px;padding:14px 16px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem;">
    <div><strong>Length:</strong> {vol['length_cm']} cm</div>
    <div> <strong>Width:</strong> {vol['width_cm']} cm</div>
    <div><strong>Volume:</strong> {vol['volume_cm3']} cm³</div>
    <div><strong>Volume:</strong> {vol['volume_liters']} L</div>
    <div><strong>Est. Weight:</strong> {vol['weight_est_g']} g</div>
    <div><strong>Est. Weight:</strong> {vol['weight_est_kg']} kg</div>
  </div>
</div>
""", unsafe_allow_html=True)

                st.markdown("### 🚦 Risk Level")

                health_label_check = result['health'].lower()
                is_fish_healthy = any(w in health_label_check for w in ["healthy", "normal", "good", "no disease", "fit"])

                if is_fish_healthy:
                    st.success("🟢 Healthy Fish")
                elif result['health_confidence'] > 80:
                    st.error("🔴 High Risk Detected")
                elif result['health_confidence'] > 60:
                    st.warning("🟡 Moderate Risk")
                else:
                    st.success("🟢 Low Risk — Likely Healthy")

        with col1:
            from PIL import ImageDraw, ImageFont

            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img, "RGBA")
            W, H = draw_img.size

            health_label = result.get("health", "") if "error" not in result else ""
            confidence   = result.get("health_confidence", 0) if "error" not in result else 0
            is_healthy   = any(w in health_label.lower() for w in ["healthy", "normal", "good", "no disease", "fit"])

            base = max(W, H)
            fs_sp = max(int(min(W, H) * 0.10), 18)
            try:
                font_sp = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs_sp)
            except:
                font_sp = ImageFont.load_default()

            cx = int(W * 0.52)
            cy = int(H * 0.46)
            max_r = min(cx, W-cx, cy, H-cy) - 30
            r = int(min(W, H) * 0.30)
            r = min(r, max_r)
            r = max(r, 20)

            if "error" not in result and not is_healthy and confidence > 0:
                if confidence > 80:   rc = (220, 38,  38)
                elif confidence > 60: rc = (234, 179, 8)
                else:                 rc = (251, 146, 60)

                for exp, alpha in [(int(r*0.18),12),(int(r*0.10),28),(int(r*0.04),50)]:
                    exp = min(exp, min(cx, W-cx, cy, H-cy) - r - 4)
                    if exp > 0:
                        draw.ellipse([cx-r-exp, cy-r-exp, cx+r+exp, cy+r+exp],
                                     outline=(*rc, alpha), width=3)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(*rc, 45), outline=(*rc, 245), width=5)
                gap = int(r * 0.25)
                draw.line([(cx, cy-r+gap), (cx, cy+r-gap)], fill=(*rc, 200), width=4)
                draw.line([(cx-r+gap, cy), (cx+r-gap, cy)], fill=(*rc, 200), width=4)

                sp_short = result.get("species", "").upper()
                sp_tw = len(sp_short) * fs_sp // 2
                sp_x = max(cx-r+6, min(cx - sp_tw//2, cx+r-sp_tw-6))
                sp_y = cy - fs_sp // 2
                draw.text((sp_x+2, sp_y+2), sp_short, fill=(0,0,0,180), font=font_sp)
                draw.text((sp_x,   sp_y),   sp_short, fill=(255,255,255,255), font=font_sp)
                caption = "🔴 Disease Region Detected"

            else:
                rc = (34, 197, 94)
                for exp, alpha in [(int(r*0.18),12),(int(r*0.10),28),(int(r*0.04),50)]:
                    exp = min(exp, min(cx, W-cx, cy, H-cy) - r - 4)
                    if exp > 0:
                        draw.ellipse([cx-r-exp, cy-r-exp, cx+r+exp, cy+r+exp],
                                     outline=(*rc, alpha), width=3)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(*rc, 45), outline=(*rc, 245), width=5)

                sp_short = result.get("species", "HEALTHY").upper() if "error" not in result else "HEALTHY"
                sp_tw = len(sp_short) * fs_sp // 2
                sp_x = max(cx-r+6, min(cx - sp_tw//2, cx+r-sp_tw-6))
                sp_y = cy - fs_sp // 2
                draw.text((sp_x+2, sp_y+2), sp_short, fill=(0,0,0,180), font=font_sp)
                draw.text((sp_x,   sp_y),   sp_short, fill=(255,255,255,255), font=font_sp)
                caption = "✅ No Disease Detected"

            st.image(draw_img, caption=caption, width='stretch')

    st.markdown("---")
    st.caption("© 2026 FishIdentification AI | Bharathiar University")

# =====================================================
# =================== SPECIES TAB =====================
# =====================================================
with tab2:
    st.title("🔬 Species Database")
    if st.button("🗄️ Load Species Data"):
        try:
            response = requests.get("http://127.0.0.1:8000/api/v1/species")
            if response.status_code == 200:
                data = response.json()
                if data:
                    st.dataframe(pd.DataFrame(data), width='stretch')
                else:
                    st.info("No species available.")
            else:
                st.error("Failed to fetch data.")
        except Exception as e:
            st.error(f"API Error: {e}")

# =====================================================
# =================== HEALTH TAB ======================
# =====================================================
with tab3:
    st.title("🩺 Health Conditions Database")
    if st.button("🗄️ Load Health Data"):
        try:
            response = requests.get("http://127.0.0.1:8000/api/v1/health_conditions")
            if response.status_code == 200:
                data = response.json()
                if data:
                    st.dataframe(pd.DataFrame(data), width='stretch')
                else:
                    st.info("No health conditions available.")
            else:
                st.error("Failed to fetch data.")
        except Exception as e:
            st.error(f"API Error: {e}")

# =====================================================
# =================== BATCH ANALYSIS TAB ==============
# =====================================================
with tab4:
    st.title("🗂️ Batch Fish Analysis")
    st.subheader("Upload multiple fish images for simultaneous analysis")
    st.markdown("---")

    batch_files = st.file_uploader(
        "📡 Upload Multiple Fish Images",
        type=["jpg", "jpeg", "png", "jfif"],
        accept_multiple_files=True
    )

    if batch_files:
        st.markdown(f"**{len(batch_files)} image(s) selected.** Click the button below to analyse all.")

        if st.button("🧬 Analyse All Images"):

            from PIL import ImageDraw, ImageFont
            import base64
            from io import BytesIO

            batch_results = []
            progress_bar  = st.progress(0, text="Analysing images...")

            for idx, bf in enumerate(batch_files):
                bf.seek(0)
                img = Image.open(bf).convert("RGB")
                bf.seek(0)

                with st.spinner(f"Analysing {bf.name}..."):
                    res = predict_image(bf)

                if "error" in res:
                    risk = "❌ Error"
                    health_label = species_label = "—"
                    species_conf = health_conf = 0
                    vol = {k: "—" for k in ["length_cm","width_cm","volume_cm3","weight_est_g","weight_est_kg"]}
                else:
                    health_label  = res.get("health", "—")
                    species_label = res.get("species", "—")
                    species_conf  = res.get("species_confidence", 0)
                    health_conf   = res.get("health_confidence", 0)
                    is_ok = any(w in health_label.lower() for w in ["healthy","normal","good","no disease","fit"])
                    risk  = ("🟢 Healthy" if is_ok else
                             "🔴 High Risk" if health_conf > 80 else
                             "🟡 Moderate" if health_conf > 60 else "🟢 Low Risk")
                    vol = estimate_volume(img)

                # Circle overlay
                draw_img2 = img.copy()
                draw = ImageDraw.Draw(draw_img2, "RGBA")
                W, H = draw_img2.size
                base = max(W, H)
                fs_sp = max(int(min(W, H) * 0.10), 18)
                try:
                    font_sp = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs_sp)
                except:
                    font_sp = ImageFont.load_default()

                cx = int(W * 0.52); cy = int(H * 0.46)
                max_r = min(cx, W-cx, cy, H-cy) - 30
                r = min(int(min(W, H) * 0.30), max_r)
                r = max(r, 20)

                is_healthy_fish = "error" not in res and any(
                    w in health_label.lower() for w in ["healthy","normal","good","no disease","fit"])

                if is_healthy_fish:
                    rc = (34, 197, 94)
                else:
                    rc = ((220,38,38) if health_conf>80 else (234,179,8) if health_conf>60 else (251,146,60))

                for exp, alpha in [(int(r*0.18),15),(int(r*0.10),30),(int(r*0.04),52)]:
                    exp = min(exp, min(cx, W-cx, cy, H-cy) - r - 4)
                    if exp > 0:
                        draw.ellipse([cx-r-exp, cy-r-exp, cx+r+exp, cy+r+exp],
                                     outline=(*rc, alpha), width=3)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(*rc, 45), outline=(*rc, 240), width=4)

                if not is_healthy_fish and "error" not in res:
                    gap = int(r * 0.25)
                    draw.line([(cx, cy-r+gap), (cx, cy+r-gap)], fill=(*rc, 190), width=3)
                    draw.line([(cx-r+gap, cy), (cx+r-gap, cy)], fill=(*rc, 190), width=3)

                sp_short = species_label.upper()
                sp_tw = len(sp_short) * fs_sp // 2
                sp_x = max(cx-r+6, min(cx - sp_tw//2, cx+r-sp_tw-6))
                sp_y = cy - fs_sp // 2
                draw.text((sp_x+2, sp_y+2), sp_short, fill=(0,0,0,180), font=font_sp)
                draw.text((sp_x,   sp_y),   sp_short, fill=(255,255,255,255), font=font_sp)

                draw_img2.thumbnail((480, 480))
                buf = BytesIO()
                draw_img2.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()

                batch_results.append({
                    "img_b64": b64, "filename": bf.name,
                    "species": species_label, "species_conf": f"{species_conf:.1f}%",
                    "health": health_label,   "health_conf":  f"{health_conf:.1f}%",
                    "risk": risk,
                    "length_cm": vol["length_cm"], "width_cm": vol["width_cm"],
                    "volume_cm3": vol["volume_cm3"], "weight_est_g": vol["weight_est_g"],
                    "weight_est_kg": vol["weight_est_kg"],
                })
                progress_bar.progress((idx+1)/len(batch_files), text=f"Analysed {idx+1} of {len(batch_files)}")

            progress_bar.empty()

            total   = len(batch_results)
            healthy = sum(1 for r in batch_results if "🟢" in r["risk"])
            high    = sum(1 for r in batch_results if "🔴" in r["risk"])
            mod     = sum(1 for r in batch_results if "🟡" in r["risk"])

            st.markdown("---")
            st.markdown("### 📊 Batch Summary")
            s1, s2, s3, s4 = st.columns(4)
            for col, val, color, label in [
                (s1, total,   "#1a8db5", "🗂️ Total Images"),
                (s2, healthy, "#16a34a", "🟢 Healthy"),
                (s3, mod,     "#d97706", "🟡 Moderate Risk"),
                (s4, high,    "#dc2626", "🔴 High Risk"),
            ]:
                with col:
                    st.markdown(f"""
<div style="background:white;border-radius:16px;padding:16px;text-align:center;
            border:1px solid rgba(30,120,160,0.12);box-shadow:0 4px 16px rgba(14,95,122,0.10);">
  <div style="font-size:2rem;font-weight:800;color:{color};">{val}</div>
  <div style="font-size:0.78rem;color:#3d6070;margin-top:2px;">{label}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🖼️ Individual Results")

            for row_start in range(0, len(batch_results), 3):
                row_items = batch_results[row_start:row_start+3]
                cols = st.columns(3)
                for col, item in zip(cols, row_items):
                    with col:
                        rc2 = "#dc2626" if "🔴" in item["risk"] else "#d97706" if "🟡" in item["risk"] else "#16a34a"
                        st.markdown(f"""
<div style="background:white;border-radius:16px;padding:14px;
            border:1px solid rgba(30,120,160,0.12);box-shadow:0 4px 16px rgba(14,95,122,0.10);margin-bottom:8px;">
  <img src="data:image/png;base64,{item['img_b64']}" style="width:100%;border-radius:10px;margin-bottom:10px;"/>
  <div style="font-size:0.72rem;color:#8aaabb;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">📁 {item['filename']}</div>
  <div style="font-size:0.82rem;margin-bottom:3px;">🔬 <strong>Species:</strong> {item['species']} <span style="color:#1a8db5;font-size:0.75rem;">({item['species_conf']})</span></div>
  <div style="font-size:0.82rem;margin-bottom:6px;">🩺 <strong>Health:</strong> {item['health']} <span style="color:#1a8db5;font-size:0.75rem;">({item['health_conf']})</span></div>
  <div style="background:rgba(14,95,122,0.07);border-radius:10px;padding:8px 10px;margin-bottom:8px;font-size:0.78rem;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;">
      <div> <strong>{item['length_cm']}</strong> cm</div>
      <div> <strong>{item['width_cm']}</strong> cm</div>
      <div> <strong>{item['volume_cm3']}</strong> cm³</div>
      <div> <strong>{item['weight_est_g']}</strong> g</div>
    </div>
  </div>
  <div style="background:{rc2};color:white;border-radius:99px;padding:4px 12px;font-size:0.78rem;font-weight:600;display:inline-block;">{item['risk']}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📋 Export Results")
            export_df = pd.DataFrame([{
                "Filename": r["filename"], "Species": r["species"],
                "Species Conf": r["species_conf"], "Health": r["health"],
                "Health Conf": r["health_conf"],
                "Risk Level": r["risk"].replace("🟢","").replace("🔴","").replace("🟡","").strip(),
                "Length (cm)": r["length_cm"], "Width (cm)": r["width_cm"],
                "Volume (cm³)": r["volume_cm3"], "Est. Weight (g)": r["weight_est_g"],
                "Est. Weight (kg)": r["weight_est_kg"],
            } for r in batch_results])
            st.dataframe(export_df, width='stretch')
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV Report", data=csv,
                               file_name="batch_fish_analysis.csv", mime="text/csv")

# =====================================================
# =================== ABOUT TAB =======================
# =====================================================
with tab5:

    st.markdown("""
<div style="font-family:'DM Sans',sans-serif;">

<div style="background:linear-gradient(135deg,rgba(14,95,122,0.07),rgba(45,212,191,0.05));
            border:1px solid rgba(30,120,160,0.15);border-radius:18px;padding:22px 26px;margin-bottom:18px;">
  <div style="font-size:0.72rem;color:#1a8db5;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:10px;">🏛️ Project Details</div>
  <div style="display:grid;gap:8px;">
    <div> <strong>Project Title:</strong> Neural Net Based Real Time Fish Catch Identification, Health &amp; Volume Estimation</div>
    <div> <strong>University:</strong> Bharathiar University, Coimbatore</div>
    <div style="margin-top:6px;"><strong>Technologies:</strong></div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:2px;">
      <span style="background:rgba(45,212,191,0.12);border:1px solid rgba(45,212,191,0.3);border-radius:99px;padding:3px 12px;font-size:0.8rem;color:#0e5f7a;">🐍 Python</span>
      <span style="background:rgba(255,119,0,0.10);border:1px solid rgba(255,119,0,0.25);border-radius:99px;padding:3px 12px;font-size:0.8rem;color:#92400e;">🧠 Deep Learning</span>
      <span style="background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.28);border-radius:99px;padding:3px 12px;font-size:0.8rem;color:#4338ca;">⚡ FastAPI</span>
      <span style="background:rgba(248,113,130,0.10);border:1px solid rgba(248,113,130,0.28);border-radius:99px;padding:3px 12px;font-size:0.8rem;color:#9f1239;">🎈 Streamlit</span>
      <span style="background:rgba(30,141,181,0.10);border:1px solid rgba(30,141,181,0.28);border-radius:99px;padding:3px 12px;font-size:0.8rem;color:#0e5f7a;">🔗 REST API</span>
      <span style="background:rgba(250,204,21,0.10);border:1px solid rgba(250,204,21,0.30);border-radius:99px;padding:3px 12px;font-size:0.8rem;color:#92400e;">🖼️ Transfer Learning</span>
    </div>
  </div>
</div>

<div style="background:linear-gradient(135deg,rgba(14,95,122,0.07),rgba(45,212,191,0.05));
            border:1px solid rgba(30,120,160,0.15);border-radius:18px;padding:22px 26px;margin-bottom:18px;">
  <div style="font-size:0.72rem;color:#1a8db5;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px;">🎯 Objectives</div>
  <div style="display:grid;gap:9px;">
    <div style="display:flex;align-items:flex-start;gap:10px;"><span style="font-size:1.2rem;"></span><span>Identify fish species in real-time using AI-powered deep learning models</span></div>
    <div style="display:flex;align-items:flex-start;gap:10px;"><span style="font-size:1.2rem;"></span><span>Detect fish health conditions and flag diseases accurately</span></div>
    <div style="display:flex;align-items:flex-start;gap:10px;"><span style="font-size:1.2rem;"></span><span>Estimate fish volume from uploaded images intelligently</span></div>
    <div style="display:flex;align-items:flex-start;gap:10px;"><span style="font-size:1.2rem;"></span><span>Support aquaculture monitoring and smart fish farm diagnosis</span></div>
    <div style="display:flex;align-items:flex-start;gap:10px;"><span style="font-size:1.2rem;"></span><span>Empower fish farmers with instant, intelligent analysis tools</span></div>
  </div>
</div>

<div style="background:linear-gradient(135deg,rgba(14,95,122,0.07),rgba(45,212,191,0.05));
            border:1px solid rgba(30,120,160,0.15);border-radius:18px;padding:22px 26px;margin-bottom:18px;">
  <div style="font-size:0.72rem;color:#1a8db5;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px;">⚡ Key Features</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">🖼️</span><span style="font-size:0.85rem;font-weight:500;">Image-based Prediction</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">🔬</span><span style="font-size:0.85rem;font-weight:500;">Species Classification</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">🦠</span><span style="font-size:0.85rem;font-weight:500;">Disease Detection</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">📈</span><span style="font-size:0.85rem;font-weight:500;">Confidence Scoring</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">🚦</span><span style="font-size:0.85rem;font-weight:500;">Risk Level Indicator</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">🗄️</span><span style="font-size:0.85rem;font-weight:500;">Database Integration</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">🎯</span><span style="font-size:0.85rem;font-weight:500;">Disease Region Overlay</span></div>
    <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.6);border-radius:10px;padding:8px 12px;border:1px solid rgba(30,120,160,0.10);"><span style="font-size:1.1rem;">⚖️</span><span style="font-size:0.85rem;font-weight:500;">Volume Estimation</span></div>
  </div>
</div>

<div style="background:linear-gradient(135deg,#0e5f7a,#1a8db5);border-radius:18px;padding:24px 26px;text-align:center;">
  <div style="font-size:2rem;margin-bottom:6px;">👩‍💻</div>
  <div style="font-size:0.72rem;color:rgba(200,235,255,0.70);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">💡 Developed By</div>
  <div style="font-size:1.25rem;font-weight:800;color:#ffffff;letter-spacing:0.5px;font-family:'Playfair Display',serif;margin-bottom:4px;">LAKSHANA .T</div>
  <div style="font-size:0.88rem;color:rgba(200,235,255,0.85);margin-bottom:2px;">🎓 Master of Artificial Intelligence</div>
  <div style="font-size:0.85rem;color:rgba(180,220,255,0.70);">🏛️ Bharathiar University, Coimbatore</div>
  <div style="margin-top:12px;font-size:0.72rem;color:rgba(180,220,255,0.50);letter-spacing:1px;">© 2026 · All Rights Reserved</div>
</div>

</div>
""", unsafe_allow_html=True)