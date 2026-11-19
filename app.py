"""
EcoSort AI — Intelligent Waste Classification and Segregation System
Professional UI: forest green, full-width, no red, large fonts, real photos.
"""

import os, sys, json, time, base64
from PIL import Image
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_waste_image, CLASSES
from src.recommendations import get_recommendation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load real photos from assets/ ──────────────────────────────────────────────
def _b64(filename: str) -> str:
    p = os.path.join(PROJECT_ROOT, "assets", filename)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    return ""

IMG = {
    "cardboard": _b64("cardboard.jpg"),
    "glass":     _b64("glass.jpg"),
    "metal":     _b64("metal.jpg"),
    "paper":     _b64("paper.jpg"),
    "plastic":   _b64("plastic.jpg"),
    "trash":     _b64("trash.jpg"),
}

# ── Master CSS — overwrites every Streamlit default ────────────────────────────
CSS = """
<style>
/* ═══════════════════════════════════════════════════
   RESET / BASE
═══════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp, .stApp > div,
#root, #root > div, .main, .main > div {
    background-color: #f5f9f6 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1e293b !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }

/* Remove all padding / margin from the outer block */
.block-container {
    max-width: 1400px !important;
    width: 96% !important;
    padding: 0 1.2rem 2rem !important;
    margin: 0 auto !important;
}

/* ═══════════════════════════════════════════════════
   BUTTONS — kill every red, enforce forest green
═══════════════════════════════════════════════════ */

/* Target ALL Streamlit buttons unconditionally */
div[data-testid="stButton"] > button,
div[data-testid="stButton"] > button:focus,
div[data-testid="stButton"] > button:active {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

/* Primary variant */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background-color: #1b4d3e !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 3px 10px rgba(27,77,62,.25) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background-color: #245e4d !important;
    box-shadow: 0 5px 14px rgba(27,77,62,.35) !important;
    transform: translateY(-1px) !important;
}

/* Secondary variant */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1.5px solid #cbd5e1 !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover {
    border-color: #1b4d3e !important;
    color: #1b4d3e !important;
    background-color: #edf6f1 !important;
}

/* Nav buttons — transparent, text-only */
.nav-wrap div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: #475569 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 0.3rem !important;
    width: 100% !important;
}
.nav-wrap div[data-testid="stButton"] > button:hover {
    color: #1b4d3e !important;
    background: transparent !important;
}
.nav-active div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid #1b4d3e !important;
    color: #1b4d3e !important;
    font-weight: 800 !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════════
   SELECT BOX / WIDGETS
═══════════════════════════════════════════════════ */
div[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    background: white !important;
}

/* ═══════════════════════════════════════════════════
   FILE UPLOADER — keep it clean
═══════════════════════════════════════════════════ */
div[data-testid="stFileUploader"] {
    border: 2px dashed #94a3b8 !important;
    border-radius: 12px !important;
    background: white !important;
    padding: 1.5rem !important;
}
div[data-testid="stFileUploader"]:hover { border-color: #1b4d3e !important; }
div[data-testid="stFileUploader"] button {
    background-color: white !important;
    border: 1.5px solid #cbd5e1 !important;
    color: #1e293b !important;
    border-radius: 6px !important;
}

/* ═══════════════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════════════ */
div[data-testid="stSpinner"] > div { border-top-color: #1b4d3e !important; }

/* ═══════════════════════════════════════════════════
   PLOTLY CHARTS
═══════════════════════════════════════════════════ */
div[data-testid="stPlotlyChart"] { background: transparent !important; }

/* ═══════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
═══════════════════════════════════════════════════ */
.tag {
    font-size: 0.8rem; font-weight: 800; color: #2d6a4f;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.4rem;
}
.h1 {
    font-size: 3rem; font-weight: 900; color: #12372c;
    line-height: 1.1; letter-spacing: -1px; margin-bottom: 0.6rem;
}
.h2 { font-size: 2.2rem; font-weight: 900; color: #12372c; letter-spacing: -.5px; }
.h3 { font-size: 1.8rem; font-weight: 800; color: #12372c; }
.lead { font-size: 1.1rem; color: #475569; line-height: 1.6; max-width: 580px; }

/* Hero right panel */
.hero-panel {
    background: #eaf4ee; border-radius: 20px;
    border: 1px solid #c3dece; padding: 1.6rem 1.2rem;
    text-align: center; box-shadow: 0 4px 20px rgba(27,77,62,.06);
}
.hero-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 0.6rem; text-align: center; width: 100px;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.hero-card img { width: 82px; height: 82px; object-fit: cover; border-radius: 8px; }
.hero-card span { display: block; font-size: 0.82rem; font-weight: 700; color: #334155; margin-top: 0.3rem; }

/* Category row cards */
.cat-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 0.9rem 1rem; display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.02); transition: all 0.15s;
}
.cat-card:hover { border-color: #2d6a4f; box-shadow: 0 4px 12px rgba(45,106,79,.1); }
.cat-card img { width: 46px; height: 46px; object-fit: cover; border-radius: 9px; }
.cat-card .info { margin-left: 0.9rem; }
.cat-card .info strong { font-size: 1.1rem; font-weight: 800; color: #0f172a; }
.cat-card .info span { font-size: 0.9rem; color: #64748b; display: block; margin-top: 1px; }
.cat-card .arrow { color: #94a3b8; font-size: 1.3rem; font-weight: 700; }

/* Dustbin guide boxes */
.bin { display: flex; align-items: flex-start; gap: 1rem;
       padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 0.6rem; }
.bin-blue  { background: #eff6ff; border: 1.5px solid #bfdbfe; }
.bin-green { background: #f0fdf4; border: 1.5px solid #bbf7d0; }
.bin-amber { background: #fefce8; border: 1.5px solid #fde68a; }
.bin-icon  { font-size: 1.8rem; flex-shrink: 0; margin-top: 2px; }
.bin-title { font-size: 1.1rem; font-weight: 800; }
.bin-desc  { font-size: 0.95rem; margin-top: 0.25rem; line-height: 1.55; }
.bin-blue  .bin-title { color: #1e40af; } .bin-blue  .bin-desc { color: #2563eb; }
.bin-green .bin-title { color: #15803d; } .bin-green .bin-desc { color: #16a34a; }
.bin-amber .bin-title { color: #854d0e; } .bin-amber .bin-desc { color: #a16207; }

/* Result card */
.result-card {
    background: white; border: 1.5px solid #e2e8f0; border-radius: 14px;
    padding: 1.3rem 1.4rem; margin-bottom: 0.8rem;
    box-shadow: 0 4px 12px rgba(0,0,0,.03);
}
.result-label { font-size: 0.82rem; font-weight: 800; color: #2d6a4f; text-transform: uppercase; letter-spacing: 1px; }
.result-class { font-size: 2.4rem; font-weight: 900; color: #12372c; line-height: 1.1; }
.result-conf  { font-size: 2.2rem; font-weight: 900; color: #16a34a; }
.result-meta  { font-size: 0.85rem; color: #64748b; font-weight: 600; margin-top: 2px; }

/* Step instructions */
.step {
    display: flex; align-items: flex-start; gap: 0.7rem;
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.55rem 0.9rem; margin-bottom: 0.4rem;
}
.step-num {
    background: #1b4d3e; color: white; border-radius: 50%;
    min-width: 22px; height: 22px; display: flex; align-items: center;
    justify-content: center; font-size: 0.72rem; font-weight: 800; flex-shrink: 0;
}
.step-txt { font-size: 0.98rem; color: #334155; line-height: 1.5; padding-top: 1px; }

/* Env tip */
.env-tip {
    background: #f0fdf4; border-left: 4px solid #16a34a;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    font-size: 0.95rem; color: #14532d; margin-top: 0.6rem;
}

/* Benchmark table */
.btable { width: 100%; border-collapse: collapse; margin-top: 0.8rem; }
.btable th {
    font-size: 0.8rem; font-weight: 800; text-transform: uppercase;
    color: #64748b; letter-spacing: 1px; padding: 0.5rem 0.9rem;
    border-bottom: 2px solid #e2e8f0; text-align: left;
}
.btable td { padding: 1rem 0.9rem; font-size: 1.05rem; font-weight: 600; border-bottom: 1px solid #f1f5f9; }
.btable tr:hover td { background: #f8fafc; }

/* Info note */
.note {
    display: flex; align-items: center; gap: 0.5rem;
    color: #64748b; font-size: 0.88rem; margin-top: 0.6rem;
}

/* Divider */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }

/* Footer */
.footer { display: flex; justify-content: space-between; align-items: center;
          color: #94a3b8; font-size: 0.88rem; padding-top: 0.6rem; }
.footer strong { color: #64748b; }

/* Badge pill */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-green { background: #dcfce7; color: #15803d; }
.badge-blue  { background: #dbeafe; color: #1d4ed8; }
.badge-amber { background: #fef9c3; color: #854d0e; }

/* Research bar */
.research-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: #f1f5f9; border-radius: 12px; padding: 0.9rem 1.2rem; margin-top: 0.8rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
if "section" not in st.session_state:
    st.session_state.section = "Home"

# ── Helpers ─────────────────────────────────────────────────────────────────────
def get_eval():
    p = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

def get_history(m: str):
    p = os.path.join(PROJECT_ROOT, "models", f"{m}_history.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

# ── Top Navigation ──────────────────────────────────────────────────────────────
st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
n0, n1, n2, n3, n4, n_cta = st.columns([2.6, 1.0, 1.7, 1.7, 0.4, 2.0])

with n0:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:.6rem;padding-top:.2rem">
      <div style="background:#1b4d3e;color:white;width:36px;height:36px;border-radius:50%;
                  display:flex;align-items:center;justify-content:center;font-size:1.1rem;font-weight:900">🌿</div>
      <span style="font-size:1.6rem;font-weight:900;color:#12372c;letter-spacing:-.5px">EcoSort</span>
      <span style="font-size:1rem;font-weight:700;color:#64748b;margin-left:-.3rem">AI</span>
    </div>""", unsafe_allow_html=True)

with n1:
    cls = "nav-active" if st.session_state.section == "Home" else "nav-wrap"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button("Home", key="n_home", use_container_width=True):
        st.session_state.section = "Home"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with n2:
    cls = "nav-active" if st.session_state.section == "Classification" else "nav-wrap"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button("Waste classification", key="n_cls", use_container_width=True):
        st.session_state.section = "Classification"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with n3:
    cls = "nav-active" if st.session_state.section == "Comparison" else "nav-wrap"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button("Model comparison", key="n_cmp", use_container_width=True):
        st.session_state.section = "Comparison"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with n_cta:
    if st.button("Try classification ↗", key="n_try", type="primary", use_container_width=True):
        st.session_state.section = "Classification"; st.rerun()

st.markdown('<hr class="divider" style="margin:.4rem 0 1.4rem">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# A.  HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.section == "Home":

    # ── Hero row ────────────────────────────────────────────────────────────
    h_l, h_r = st.columns([1.3, 1.0], gap="large")

    with h_l:
        st.markdown("""
        <div class="tag">● Your everyday guide to better sorting</div>
        <div class="h1">Small choices.<br>A cleaner future.</div>
        <div class="lead">
          Find the right next step for everyday waste. Learn what belongs in the
          blue bin, what needs a closer look, and how image recognition is being explored.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        b1, b2 = st.columns([1.4, 1.0])
        with b1:
            if st.button("Explore waste classification →", type="primary", key="h_cls", use_container_width=True):
                st.session_state.section = "Classification"; st.rerun()
        with b2:
            if st.button("Our research  >", key="h_res", use_container_width=True):
                st.session_state.section = "Comparison"; st.rerun()

        st.markdown("""
        <div style="display:flex;align-items:center;gap:.8rem;background:#eaf4ee;
                    border:1px solid #c3dece;border-radius:12px;padding:.7rem 1rem;
                    margin-top:1.2rem;max-width:360px">
          <span style="font-size:1.3rem">🛡️</span>
          <div>
            <div style="font-weight:800;font-size:1rem;color:#12372c">Thoughtful by design</div>
            <div style="font-size:.9rem;color:#475569">Clear guidance, no guesswork</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with h_r:
        p_i  = IMG["paper"]
        pl_i = IMG["plastic"]
        g_i  = IMG["glass"]
        st.markdown(f"""
        <div class="hero-panel">
          <div style="display:inline-block;border:1.5px dashed #2d6a4f;border-radius:50px;
                      padding:3px 14px;font-size:.72rem;font-weight:800;color:#1b4d3e;
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem">
            Sort with care
          </div>
          <div style="display:flex;justify-content:center;gap:1rem;margin:.4rem 0">
            <div class="hero-card">
              <img src="{p_i}" alt="paper"><span>paper</span>
            </div>
            <div class="hero-card" style="border:2px solid #3b82f6;box-shadow:0 4px 12px rgba(59,130,246,.2)">
              <img src="{pl_i}" alt="plastic"><span style="color:#1d4ed8">plastic</span>
            </div>
            <div class="hero-card" style="border:2px solid #16a34a;box-shadow:0 4px 12px rgba(22,163,74,.15)">
              <img src="{g_i}" alt="glass"><span style="color:#15803d">glass</span>
            </div>
          </div>
          <div style="font-weight:800;font-size:1.05rem;color:#1b4d3e;margin-top:1rem">
            A better habit starts here.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Know-what-goes-where ────────────────────────────────────────────────
    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:.8rem">
      <div>
        <div class="tag">A little more clarity</div>
        <div class="h2">Know what goes where.</div>
      </div>
      <div style="color:#64748b;font-size:.95rem;font-weight:600;text-align:right">
        Six everyday categories.<br>One more considered routine.
      </div>
    </div>
    """, unsafe_allow_html=True)

    CATS = [
        ("cardboard", "Cardboard", "Dry · Blue bin"),
        ("glass",     "Glass",     "Dry · Blue bin"),
        ("metal",     "Metal",     "Dry · Blue bin"),
        ("paper",     "Paper",     "Dry · Blue bin"),
        ("plastic",   "Plastic",   "Dry · Blue bin"),
        ("trash",     "Trash",     "Please inspect"),
    ]
    r1, r2, r3 = st.columns(3)
    for i, (key, label, sub) in enumerate(CATS):
        col = [r1, r2, r3][i % 3]
        with col:
            st.markdown(f"""
            <div class="cat-card">
              <div style="display:flex;align-items:center">
                <img src="{IMG[key]}" alt="{label}"
                     style="width:46px;height:46px;object-fit:cover;border-radius:9px;border:1px solid #e2e8f0">
                <div style="margin-left:.9rem">
                  <strong style="font-size:1.12rem;font-weight:800;color:#0f172a">{label}</strong>
                  <span style="font-size:.9rem;color:#64748b;display:block;margin-top:1px">{sub}</span>
                </div>
              </div>
              <span style="color:#94a3b8;font-size:1.3rem;font-weight:700">›</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note">
      <span>ⓘ</span>
      <span>Local recycling rules may vary. Check your municipality's guidance for accepted materials.</span>
    </div>""", unsafe_allow_html=True)

    # ── Research strip ──────────────────────────────────────────────────────
    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    r_l, r_r = st.columns([4, 1.2])
    with r_l:
        st.markdown("""
        <div class="research-bar">
          <div>
            <div style="font-size:.78rem;font-weight:800;color:#2d6a4f;text-transform:uppercase;letter-spacing:1px">A research project</div>
            <div style="font-size:1.3rem;font-weight:900;color:#12372c;margin-top:.15rem">Three models. Real results.</div>
            <div style="font-size:.95rem;color:#475569;margin-top:.2rem">
              Explore how a custom CNN, ResNet-18 and DenseNet-121 compare when trained and evaluated on the same waste dataset.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    with r_r:
        st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
        if st.button("Explore the models →", key="h_cmp2", use_container_width=True):
            st.session_state.section = "Comparison"; st.rerun()

    st.markdown('<hr class="divider" style="margin:1.4rem 0 .6rem">', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
      <strong>EcoSort AI · Smart Waste Recognition. Responsible Disposal.</strong>
      <span>Check your local sorting rules.</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# B.  WASTE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.section == "Classification":
    st.markdown("""
    <div style="font-size:.92rem;color:#64748b;margin-bottom:.4rem">
      Home &nbsp;›&nbsp; <strong style="color:#12372c">Waste classification</strong>
    </div>
    <div class="tag">A closer look at your waste</div>
    <div class="h2" style="margin-bottom:.25rem">What are we sorting?</div>
    <div style="font-size:1.05rem;color:#475569;margin-bottom:1.2rem">
      Upload one clear photo to explore the classification workflow and disposal guidance.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown('<div style="font-size:.88rem;font-weight:800;color:#1b4d3e;text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem">01 &nbsp; Add a photo</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "JPG, JPEG or PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        # Quick sample selector
        sample_opt = st.selectbox(
            "Or pick a dataset sample:",
            ["— select sample —", "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"],
            label_visibility="collapsed",
        )

        pil_img = None
        src_label = ""
        if uploaded is not None:
            pil_img = Image.open(uploaded)
            src_label = "Uploaded image"
        elif sample_opt != "— select sample —":
            sp = os.path.join(PROJECT_ROOT, "assets", f"{sample_opt.lower()}.jpg")
            if os.path.exists(sp):
                pil_img = Image.open(sp)
                src_label = f"Dataset sample – {sample_opt}"

        if pil_img is not None:
            st.image(pil_img, caption=src_label, width=230)

        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.88rem;font-weight:800;color:#1b4d3e;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem">02 &nbsp; Select a model</div>', unsafe_allow_html=True)

        model_sel = st.selectbox(
            "Model",
            ["DenseNet-121 (Best accuracy)", "ResNet-18", "Custom CNN"],
            label_visibility="collapsed",
        )
        model_map = {
            "DenseNet-121 (Best accuracy)": "densenet121",
            "ResNet-18":                    "resnet18",
            "Custom CNN":                   "custom_cnn",
        }
        chosen = model_map[model_sel]

        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        run_btn = st.button("Check model availability →", type="primary", key="run_cls", use_container_width=True)
        st.markdown("""
        <div class="note" style="margin-top:.5rem">
          <span>⚠</span>
          <span style="font-style:italic;color:#94a3b8">
            The image-upload preview works independently; model results require a genuinely trained checkpoint.
          </span>
        </div>""", unsafe_allow_html=True)

    with right:
        if not run_btn and "live_pred" not in st.session_state:
            # Default guide panel
            st.markdown("""
            <div class="tag" style="margin-bottom:.4rem">♻ &nbsp; Two-bin sorting guide</div>
            <div class="h3" style="margin-bottom:1rem">Start with what you know.</div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="bin bin-blue">
              <div class="bin-icon">🔵</div>
              <div>
                <div class="bin-title">Blue · dry waste</div>
                <div class="bin-desc">Cardboard, glass, metal, paper and plastic—if accepted locally.</div>
              </div>
            </div>
            <div class="bin bin-green">
              <div class="bin-icon">🟢</div>
              <div>
                <div class="bin-title">Green · wet waste</div>
                <div class="bin-desc">Biodegradable food scraps such as vegetable and fruit peels.</div>
              </div>
            </div>
            <div class="bin bin-amber" style="margin-top:.6rem">
              <div class="bin-icon">⚠️</div>
              <div>
                <div class="bin-title">Trash needs a closer look.</div>
                <div class="bin-desc">
                  This model has no wet waste class. Inspect the item before choosing a bin;
                  hazardous and special waste needs its own local guidance.
                </div>
              </div>
            </div>
            <div class="note" style="margin-top:.8rem">
              <span>ⓘ</span> <span>Local rules always take priority.</span>
            </div>""", unsafe_allow_html=True)
        else:
            if pil_img is None:
                st.warning("Please upload an image or choose a dataset sample first.")
            else:
                with st.spinner(f"Running inference — {chosen} …"):
                    try:
                        t0 = time.time()
                        pred = predict_waste_image(pil_img, model_name=chosen)
                        lat  = (time.time() - t0) * 1000
                        st.session_state["live_pred"] = pred
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        st.stop()

                p_cls  = pred["predicted_class"]
                p_title = pred["predicted_title"]
                conf   = pred["confidence"]
                probs  = pred["class_probabilities"]

                # Result header
                st.markdown(f"""
                <div class="result-card">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <div class="result-label">Predicted category</div>
                      <div class="result-class">{p_title}</div>
                    </div>
                    <div style="text-align:right">
                      <div class="result-conf">{conf:.1f}%</div>
                      <div class="result-meta">{chosen.upper()} · {lat:.1f} ms</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Dustbin recommendation
                if pred.get("verification_required"):
                    st.markdown("""
                    <div class="bin bin-amber">
                      <div class="bin-icon">⚠️</div>
                      <div>
                        <div class="bin-title">User verification required</div>
                        <div class="bin-desc">
                          Classified as residual waste (Trash).<br>
                          • Biodegradable organic → <strong style="color:#15803d">Green bin</strong><br>
                          • Dry non-recyclable → Designated dry residual collection<br>
                          • Hazardous / e-waste → Separate municipal collection
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bin bin-blue">
                      <div class="bin-icon">🔵</div>
                      <div>
                        <div class="bin-title">{pred.get('dustbin','Blue Bin').upper()}</div>
                        <div class="bin-desc">Place this clean, dry recyclable item in the blue dry-waste bin.</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                # Probability chart
                c_names = [c.capitalize() for c in CLASSES]
                c_vals  = [probs[c] for c in CLASSES]
                b_colors = ["#1b4d3e" if c.lower() == p_cls else "#cbd5e1" for c in CLASSES]
                fig = go.Figure(go.Bar(
                    x=c_vals, y=c_names, orientation="h",
                    marker_color=b_colors,
                    text=[f"{v:.1f}%" for v in c_vals], textposition="outside",
                ))
                fig.update_layout(
                    title=dict(text="Class probabilities", font=dict(size=13, color="#12372c")),
                    xaxis_range=[0, max(c_vals) * 1.3 + 5],
                    height=200, margin=dict(l=5, r=35, t=28, b=5),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False),
                    font=dict(family="Inter", size=12),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Disposal steps
                st.markdown('<div style="font-size:1rem;font-weight:800;color:#12372c;margin:.5rem 0 .3rem">Disposal steps:</div>', unsafe_allow_html=True)
                for i, step in enumerate(pred.get("instructions", []), 1):
                    st.markdown(f"""
                    <div class="step">
                      <div class="step-num">{i}</div>
                      <div class="step-txt">{step}</div>
                    </div>""", unsafe_allow_html=True)

                # Environmental tip
                tip = pred.get("environmental_tip", "")
                if tip:
                    st.markdown(f'<div class="env-tip">🌱 <strong>Environmental tip:</strong> {tip}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider" style="margin:1.6rem 0 .6rem">', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
      <strong>EcoSort AI · Smart Waste Recognition. Responsible Disposal.</strong>
      <span>Check your local sorting rules.</span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# C.  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.section == "Comparison":
    st.markdown("""
    <div style="font-size:.92rem;color:#64748b;margin-bottom:.4rem">
      Home &nbsp;›&nbsp; <strong style="color:#12372c">Model comparison</strong>
    </div>
    <div class="tag">The research, without the guesswork</div>
    <div class="h2" style="margin-bottom:.25rem">Model comparison</div>
    <div style="font-size:1.05rem;color:#475569;margin-bottom:1.2rem">
      Three architectures, evaluated on the same held-out test set.
    </div>
    """, unsafe_allow_html=True)

    eval_data = get_eval()

    if eval_data is None or "models" not in eval_data:
        # No results yet — clean placeholder
        st.markdown("""
        <div style="background:white;border:1.5px solid #e2e8f0;border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1.2rem">
          <div style="display:flex;align-items:center;gap:.9rem">
            <span style="font-size:1.6rem">🧪</span>
            <div>
              <div style="font-size:.82rem;font-weight:800;color:#d97706;text-transform:uppercase;letter-spacing:1px">
                Results not yet available
              </div>
              <div style="font-size:1.15rem;font-weight:800;color:#1e293b;margin-top:.2rem">
                Evaluation starts with trained models.
              </div>
              <div style="font-size:.95rem;color:#64748b;margin-top:.3rem">
                No model checkpoints or test evaluation report were supplied.
                Accuracy, precision, recall, F1, inference time, and confusion
                matrices will appear here only after a genuine evaluation run.
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        rows = [
            ("01", "Custom CNN",    "#0284c7"),
            ("02", "ResNet-18",     "#16a34a"),
            ("03", "DenseNet-121",  "#d97706"),
        ]
        tbl = '<table class="btable"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead><tbody>'
        for num, name, color in rows:
            tbl += f'<tr><td><strong style="color:{color}">{num} &nbsp; {name}</strong></td><td style="color:#94a3b8">Not evaluated</td><td style="color:#94a3b8">Not evaluated</td><td style="color:#94a3b8">Not evaluated</td><td style="color:#94a3b8">Not evaluated</td></tr>'
        tbl += '</tbody></table>'
        st.markdown(tbl, unsafe_allow_html=True)

        st.markdown("""
        <div class="note" style="margin-top:.6rem">
          <span>ⓘ</span> <span>No sample values or simulated performance results are shown.</span>
        </div>""", unsafe_allow_html=True)

    else:
        md = eval_data["models"]
        best = eval_data.get("best_accuracy_model", "densenet121")

        # Banner
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:12px;
                    padding:1rem 1.4rem;margin-bottom:1.2rem;
                    display:flex;align-items:center;justify-content:space-between">
          <div style="display:flex;align-items:center;gap:.9rem">
            <span style="font-size:1.5rem;background:#dcfce7;padding:.5rem;border-radius:50%">🧪</span>
            <div>
              <div style="font-size:.8rem;font-weight:800;color:#15803d;text-transform:uppercase;letter-spacing:1px">Empirical test results available</div>
              <div style="font-size:1.1rem;font-weight:800;color:#14532d">
                Evaluated on {eval_data.get('test_samples',600)} verified test images across all six classes.
              </div>
            </div>
          </div>
          <div style="text-align:right">
            <div style="font-size:.82rem;color:#166534;font-weight:700">Top architecture</div>
            <div style="font-size:1.3rem;font-weight:900;color:#14532d">🏆 {best.upper()} ({md[best]['accuracy']:.2f}%)</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Metrics table
        rows = [
            ("custom_cnn",   "01", "Custom CNN",   "#0284c7"),
            ("resnet18",     "02", "ResNet-18",    "#16a34a"),
            ("densenet121",  "03", "DenseNet-121", "#d97706"),
        ]
        tbl = '<table class="btable"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead><tbody>'
        for key, num, name, color in rows:
            m = md.get(key, {})
            tbl += (f'<tr>'
                    f'<td><strong style="color:{color}">{num} &nbsp; {name}</strong></td>'
                    f'<td>{m.get("accuracy",0):.2f}%</td>'
                    f'<td>{m.get("precision_macro",0):.2f}%</td>'
                    f'<td>{m.get("recall_macro",0):.2f}%</td>'
                    f'<td><strong style="color:#12372c">{m.get("f1_macro",0):.2f}%</strong></td>'
                    f'</tr>')
        tbl += '</tbody></table>'
        st.markdown(tbl, unsafe_allow_html=True)

        st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
        btn_c, _ = st.columns([1.6, 3.4])
        with btn_c:
            if st.button("Go to waste classification →", key="cmp_to_cls", type="primary", use_container_width=True):
                st.session_state.section = "Classification"; st.rerun()

        # Tabs for charts
        st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["📊 Metric charts", "📈 Training curves", "🎯 Confusion matrices"])

        mkeys   = ["custom_cnn", "resnet18", "densenet121"]
        mlabels = ["Custom CNN", "ResNet-18", "DenseNet-121"]

        with t1:
            c1, c2 = st.columns([1.6, 1.0])
            with c1:
                fig_b = go.Figure()
                for metric, color in [("accuracy","#1b4d3e"),("precision_macro","#2d6a4f"),
                                       ("recall_macro","#52b788"),("f1_macro","#74c69d")]:
                    fig_b.add_trace(go.Bar(
                        name=metric.replace("_macro","").capitalize(),
                        x=mlabels, y=[md[k][metric] for k in mkeys],
                        marker_color=color))
                fig_b.update_layout(
                    title="Metrics comparison (%)", barmode="group", yaxis_range=[0,105],
                    height=290, margin=dict(l=5,r=5,t=30,b=5),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter",size=11))
                st.plotly_chart(fig_b, use_container_width=True)
            with c2:
                lats = [md[k]["inference_time_ms"] for k in mkeys]
                fig_lat = go.Figure(go.Bar(
                    x=mlabels, y=lats,
                    marker_color=["#0284c7","#16a34a","#d97706"],
                    text=[f"{l:.1f} ms" for l in lats], textposition="outside"))
                fig_lat.update_layout(
                    title="Latency (ms/sample)", height=290,
                    margin=dict(l=5,r=5,t=30,b=5),
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter",size=11))
                st.plotly_chart(fig_lat, use_container_width=True)

        with t2:
            tc1, tc2 = st.columns(2)
            histories = {n: get_history(k) for k, n in zip(mkeys, mlabels)}
            palette = {"Custom CNN":"#e63946","ResNet-18":"#1d4ed8","DenseNet-121":"#10b981"}
            with tc1:
                fl = go.Figure()
                for n, h in histories.items():
                    if h: fl.add_trace(go.Scatter(x=h["epochs"],y=h["val_loss"],mode="lines+markers",name=n,line=dict(color=palette[n],width=2)))
                fl.update_layout(title="Validation loss",height=270,margin=dict(l=5,r=5,t=30,b=5),font=dict(family="Inter",size=11))
                st.plotly_chart(fl, use_container_width=True)
            with tc2:
                fa = go.Figure()
                for n, h in histories.items():
                    if h: fa.add_trace(go.Scatter(x=h["epochs"],y=h["val_acc"],mode="lines+markers",name=n,line=dict(color=palette[n],width=2)))
                fa.update_layout(title="Validation accuracy (%)",height=270,margin=dict(l=5,r=5,t=30,b=5),font=dict(family="Inter",size=11))
                st.plotly_chart(fa, use_container_width=True)

        with t3:
            cm1, cm2, cm3 = st.columns(3)
            for col, key in zip([cm1, cm2, cm3], ["densenet121","resnet18","custom_cnn"]):
                with col:
                    if key in md and "confusion_matrix" in md[key]:
                        cm  = np.array(md[key]["confusion_matrix"])
                        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1)[:,None], 1) * 100
                        fig_cm = px.imshow(
                            cmn, text_auto=".0f",
                            labels=dict(x="Predicted",y="True",color="%"),
                            x=[c.capitalize() for c in CLASSES],
                            y=[c.capitalize() for c in CLASSES],
                            color_continuous_scale="Greens",
                        )
                        fig_cm.update_layout(
                            title=key.upper(), height=280,
                            margin=dict(l=5,r=5,t=30,b=5), font=dict(size=10))
                        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<hr class="divider" style="margin:1.4rem 0 .6rem">', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
      <strong>EcoSort AI · Smart Waste Recognition. Responsible Disposal.</strong>
      <span>Check your local sorting rules.</span>
    </div>""", unsafe_allow_html=True)
