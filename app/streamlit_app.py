import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import time
import datetime
import base64
from io import BytesIO

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="SteelGuard AI · Tata Steel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_DISPLAY = {
    "crazing":          "Crazing",
    "inclusion":        "Inclusion",
    "patches":          "Patches",
    "pitted_surface":   "Pitted Surface",
    "rolled-in_scale":  "Rolled-in Scale",
    "scratches":        "Scratches",
}

# Risk tier for each defect class
RISK_MAP = {
    "scratches":        ("HIGH",   "#e74c3c", "#1a0606", "#4a1515"),
    "crazing":          ("HIGH",   "#e74c3c", "#1a0606", "#4a1515"),
    "rolled-in_scale":  ("MEDIUM", "#f39c12", "#1a1206", "#4a3010"),
    "pitted_surface":   ("MEDIUM", "#f39c12", "#1a1206", "#4a3010"),
    "inclusion":        ("LOW",    "#27ae60", "#061a0e", "#1a4028"),
    "patches":          ("LOW",    "#27ae60", "#061a0e", "#1a4028"),
}

# Colour accent per class for the bar chart
CLASS_COLORS = {
    "crazing":          "#e74c3c",
    "inclusion":        "#3498db",
    "patches":          "#2ecc71",
    "pitted_surface":   "#f39c12",
    "rolled-in_scale":  "#9b59b6",
    "scratches":        "#e67e22",
}

# Defect descriptions shown in the result card
CLASS_DESC = {
    "crazing":          "Network of fine surface cracks forming a web-like pattern.",
    "inclusion":        "Foreign particles embedded during the rolling process.",
    "patches":          "Irregular discoloured or rough surface regions.",
    "pitted_surface":   "Small cavities or craters on the steel surface.",
    "rolled-in_scale":  "Oxide scale pressed into the surface during hot-rolling.",
    "scratches":        "Linear marks caused by abrasion during handling or transport.",
}

# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main  { background: #080b11; }
.block-container { padding: 1.2rem 2rem 3rem; max-width: 1440px; }

/* ── SIDEBAR ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0a0d14;
    border-right: 1px solid #161c28;
}
section[data-testid="stSidebar"] label { color: #606880 !important; font-size: 0.8rem; }

/* ── HEADER ─────────────────────────────────── */
.hdr {
    display:flex; align-items:center; gap:16px;
    background:#0d1120; border:1px solid #1a2035;
    border-top:2px solid #c0392b;
    border-radius:0 0 12px 12px;
    padding:18px 24px; margin-bottom:22px;
}
.hdr-icon {
    width:46px; height:46px;
    background:linear-gradient(140deg,#9b1c1c,#e74c3c);
    border-radius:10px; font-size:22px;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; box-shadow:0 4px 18px rgba(231,76,60,.3);
}
.hdr-name { font-size:1.45rem; font-weight:600; color:#edf0f7; letter-spacing:-.01em; }
.hdr-sub  { font-size:.7rem; color:#3d4560;
            font-family:'IBM Plex Mono',monospace;
            letter-spacing:.07em; margin-top:2px; }
.hdr-right { margin-left:auto; display:flex; align-items:center; gap:12px; }
.badge { padding:5px 13px; border-radius:20px;
         font-size:.67rem; font-family:'IBM Plex Mono',monospace;
         letter-spacing:.08em; font-weight:500; }
.badge-online { background:#061a0e; border:1px solid #1a4028; color:#27ae60; }
.badge-time   { background:#0d1120; border:1px solid #1a2035; color:#3d4560; }
.dot { width:7px; height:7px; background:#27ae60; border-radius:50%;
       display:inline-block; margin-right:5px; animation:blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── SCAN ANIMATION ─────────────────────────── */
@keyframes scan { 0%{top:0;opacity:.8} 100%{top:100%;opacity:.05} }
.scan-wrap { position:relative; overflow:hidden; border-radius:8px; }
.scan-wrap img { display:block; width:100%; }
.scan-line {
    position:absolute; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,#e74c3c88,#e74c3c,#e74c3c88,transparent);
    animation:scan 1.8s linear infinite; pointer-events:none;
}

/* ── PANELS ─────────────────────────────────── */
.panel {
    background:#0d1120; border:1px solid #1a2035;
    border-radius:12px; padding:20px; margin-bottom:16px;
}
.panel-title {
    font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
    color:#3a4260; font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid #131826; padding-bottom:9px; margin-bottom:16px;
}

/* ── DEFECT VERDICT ─────────────────────────── */
.defect-chip {
    display:inline-flex; align-items:center; gap:8px;
    padding:7px 14px; border-radius:8px;
    font-size:.95rem; font-weight:600; margin-bottom:10px;
}
.defect-desc {
    font-size:.8rem; color:#5c6680; line-height:1.6;
    margin-bottom:16px; padding-bottom:14px;
    border-bottom:1px solid #131826;
}

/* ── GAUGE ──────────────────────────────────── */
.gauge-wrap { display:flex; flex-direction:column; align-items:center; margin:6px 0 4px; }
.gauge-num  {
    font-size:2.5rem; font-weight:500; color:#edf0f7;
    font-family:'IBM Plex Mono',monospace;
    letter-spacing:-.04em; margin-top:-10px;
}
.gauge-lbl { font-size:.62rem; letter-spacing:.12em; color:#3a4260; text-transform:uppercase; }

/* ── PROGRESS ───────────────────────────────── */
.prog { background:#131826; border-radius:3px; height:4px; width:100%; overflow:hidden; margin:8px 0 16px; }
.prog-bar { height:4px; border-radius:3px; }

/* ── METRIC TILES ───────────────────────────── */
.tiles { display:flex; gap:10px; }
.tile  {
    flex:1; background:#080b11; border:1px solid #161c28;
    border-radius:10px; padding:12px 14px;
}
.tile-lbl { font-size:.58rem; letter-spacing:.12em; text-transform:uppercase;
            color:#3a4260; font-family:'IBM Plex Mono',monospace; margin-bottom:5px; }
.tile-val { font-size:1.05rem; font-weight:600;
            color:#b0b8cc; font-family:'IBM Plex Mono',monospace; }

/* ── PROB BAR CHART ─────────────────────────── */
.prob-row {
    display:flex; align-items:center; gap:10px;
    margin-bottom:9px;
}
.prob-name {
    font-size:.72rem; color:#8892a4;
    font-family:'IBM Plex Mono',monospace;
    width:130px; flex-shrink:0;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.prob-track {
    flex:1; background:#131826; border-radius:3px;
    height:8px; overflow:hidden;
}
.prob-fill { height:8px; border-radius:3px; transition:width .5s ease; }
.prob-pct  {
    font-size:.7rem; color:#3a4260;
    font-family:'IBM Plex Mono',monospace;
    width:44px; text-align:right; flex-shrink:0;
}
.prob-row.active .prob-name { color:#edf0f7; font-weight:500; }
.prob-row.active .prob-pct  { color:#edf0f7; }

/* ── HEATMAP LEGEND ─────────────────────────── */
.legend { display:flex; align-items:center; gap:6px; margin-top:8px;
          font-size:.6rem; font-family:'IBM Plex Mono',monospace; color:#3a4260; }
.lgd-grey    { flex:1; height:5px; border-radius:3px; background:linear-gradient(90deg,#000,#fff); }
.lgd-inferno { flex:1; height:5px; border-radius:3px;
               background:linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#f0f921); }
.lgd-jet     { flex:1; height:5px; border-radius:3px;
               background:linear-gradient(90deg,#000080,#00ffff,#ffff00,#ff0000); }

/* ── HISTORY ─────────────────────────────────── */
.hist-row {
    display:flex; align-items:center; gap:10px;
    padding:8px 12px; border-radius:8px;
    border:1px solid #131826; background:#080b11;
    margin-bottom:5px;
}
.hdot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.hname { color:#8892a4; flex:1; font-family:'IBM Plex Mono',monospace;
         overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.67rem; }
.hlabel { font-family:'IBM Plex Mono',monospace; font-size:.67rem; color:#5c6680; }
.hconf { color:#3a4260; font-family:'IBM Plex Mono',monospace;
         min-width:44px; text-align:right; font-size:.67rem; }
.htime { color:#222838; font-family:'IBM Plex Mono',monospace; font-size:.6rem; }

/* ── MISC ────────────────────────────────────── */
.sec-lbl {
    font-size:.62rem; letter-spacing:.15em; text-transform:uppercase;
    color:#3a4260; font-family:'IBM Plex Mono',monospace; margin:18px 0 10px;
}
[data-testid="stFileUploader"] {
    border:1.5px dashed #1a2035 !important;
    border-radius:10px !important; background:#0a0d14 !important;
}
.stDownloadButton > button {
    background:#0d1120 !important; border:1px solid #1a2035 !important;
    color:#e74c3c !important; font-family:'IBM Plex Mono',monospace !important;
    font-size:.72rem !important; letter-spacing:.06em !important;
    border-radius:8px !important; padding:8px 20px !important; transition:all .2s;
}
.stDownloadButton > button:hover {
    background:#e74c3c !important; color:#fff !important; border-color:#e74c3c !important;
}
.empty-state {
    background:#0a0d14; border:1.5px dashed #161c28;
    border-radius:12px; padding:60px 32px; text-align:center;
}
.empty-icon { font-size:2.4rem; margin-bottom:12px; }
.empty-text { font-size:.85rem; line-height:1.8; color:#3a4260; }
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:#080b11}
::-webkit-scrollbar-thumb{background:#161c28;border-radius:4px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def svg_gauge(pct: float, color: str) -> str:
    r = 66; circ = 3.14159 * r; dash = circ * pct
    # derive a dark track from the accent colour
    return f"""<svg width="172" height="92" viewBox="0 0 172 92">
  <path d="M20,86 A66,66 0 0,1 152,86"
        fill="none" stroke="#131826" stroke-width="9" stroke-linecap="round"/>
  <path d="M20,86 A66,66 0 0,1 152,86"
        fill="none" stroke="{color}" stroke-width="9" stroke-linecap="round"
        stroke-dasharray="{dash:.1f} {circ:.1f}"/>
</svg>"""

def img_to_b64(img_rgb: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def risk_badge_html(risk: str, color: str, bg: str, border: str) -> str:
    return (f'<span style="background:{bg};border:1px solid {border};'
            f'color:{color};padding:4px 12px;border-radius:20px;'
            f'font-size:.68rem;font-family:\'IBM Plex Mono\',monospace;'
            f'letter-spacing:.08em;font-weight:600">{risk}</span>')

def prob_bar_html(probs: np.ndarray, pred_idx: int) -> str:
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        p   = float(probs[i])
        pct = p * 100
        col = CLASS_COLORS[cls]
        active = "active" if i == pred_idx else ""
        rows.append(f"""
<div class="prob-row {active}">
  <div class="prob-name">{CLASS_DISPLAY[cls]}</div>
  <div class="prob-track">
    <div class="prob-fill" style="width:{pct:.1f}%;background:{col}"></div>
  </div>
  <div class="prob-pct">{pct:.1f}%</div>
</div>""")
    return "".join(rows)

# ══════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════
@st.cache_resource
def load_defect_model():
    return load_model("models/best_model.keras")

model = load_defect_model()

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
now_str = datetime.datetime.now().strftime("%H:%M:%S")

st.sidebar.markdown(
    '<div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;'
    'color:#3a4260;font-family:\'IBM Plex Mono\',monospace;padding:4px 0 12px">'
    'Model Controls</div>',
    unsafe_allow_html=True,
)

show_conf    = st.sidebar.checkbox("Show Confidence Gauge", True)
show_heatmap = st.sidebar.checkbox("Show Heatmap Panels", True)
show_hist    = st.sidebar.checkbox("Show Inspection History", True)

st.sidebar.markdown("<hr style='border-color:#161c28;margin:16px 0'>", unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div style="font-size:.62rem;color:#28304a;font-family:\'IBM Plex Mono\',monospace;line-height:2.1">'
    f'MODEL &nbsp;&nbsp;&nbsp;&nbsp;ResNet-50 v2<br>'
    f'TASK &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6-Class Defect Detection<br>'
    f'BACKBONE &nbsp;ImageNet → NEU-DET<br>'
    f'INPUT &nbsp;&nbsp;&nbsp;&nbsp;224 × 224 × 3<br>'
    f'CLASSES &nbsp;&nbsp;6 defect types<br>'
    f'CLOCK &nbsp;&nbsp;&nbsp;&nbsp;{now_str}'
    f'</div>',
    unsafe_allow_html=True,
)

# Class legend in sidebar
st.sidebar.markdown("<hr style='border-color:#161c28;margin:16px 0'>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;'
    'color:#3a4260;font-family:\'IBM Plex Mono\',monospace;margin-bottom:10px">'
    'Defect Classes</div>',
    unsafe_allow_html=True,
)
for cls in CLASS_NAMES:
    risk, col, bg, border = RISK_MAP[cls]
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        f'<div style="width:8px;height:8px;background:{col};border-radius:50%;flex-shrink:0"></div>'
        f'<span style="font-size:.7rem;color:#8892a4;font-family:\'IBM Plex Mono\',monospace;flex:1">'
        f'{CLASS_DISPLAY[cls]}</span>'
        f'<span style="font-size:.6rem;color:{col};font-family:\'IBM Plex Mono\',monospace">{risk}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Inspection history
if show_hist and st.session_state.history:
    st.sidebar.markdown("<hr style='border-color:#161c28;margin:16px 0'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;'
        'color:#3a4260;font-family:\'IBM Plex Mono\',monospace;margin-bottom:10px">'
        'Recent Inspections</div>',
        unsafe_allow_html=True,
    )
    for h in reversed(st.session_state.history[-7:]):
        col = CLASS_COLORS.get(h["label"], "#888")
        st.sidebar.markdown(
            f'<div class="hist-row">'
            f'<div class="hdot" style="background:{col}"></div>'
            f'<div class="hname">{h["name"]}</div>'
            f'<div class="hlabel">{CLASS_DISPLAY.get(h["label"],h["label"])}</div>'
            f'<div class="hconf">{h["conf"]*100:.1f}%</div>'
            f'<div class="htime">{h["time"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
now_full = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(
    f'<div class="hdr">'
    f'<div class="hdr-icon">🛡️</div>'
    f'<div>'
    f'<div class="hdr-name">SteelGuard AI</div>'
    f'<div class="hdr-sub">SURFACE DEFECT DETECTION SYSTEM · TATA STEEL QC DIVISION</div>'
    f'</div>'
    f'<div class="hdr-right">'
    f'<div class="badge badge-time">{now_full}</div>'
    f'<div class="badge badge-online"><span class="dot"></span>SYSTEM ONLINE</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════
st.markdown('<div class="sec-lbl">Surface Image Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop steel surface image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if uploaded_file is not None:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_resized = cv2.resize(img_np, (224, 224)) / 255.0
    img_input   = np.expand_dims(img_resized, axis=0)

    with st.spinner("Running AI inference…"):
        time.sleep(0.5)
        raw = model.predict(img_input, verbose=0)[0]

    # If binary model, expand to dummy 6-class for compatibility
    if raw.shape[0] == 1:
        # binary: treat output as "scratches" probability
        p_crack = float(raw[0])
        probs   = np.zeros(6, dtype=np.float32)
        # distribute: scratches if crack, else patches
        if p_crack > 0.5:
            probs[5] = p_crack          # scratches
            probs[2] = 1.0 - p_crack
        else:
            probs[2] = 1.0 - p_crack   # patches
            probs[5] = p_crack
    else:
        probs = raw

    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    pct        = int(confidence * 100)

    risk_str, risk_color, risk_bg, risk_border = RISK_MAP[pred_label]
    chip_color = CLASS_COLORS[pred_label]

    # log history
    st.session_state.history.append({
        "name":  uploaded_file.name,
        "label": pred_label,
        "conf":  confidence,
        "time":  datetime.datetime.now().strftime("%H:%M:%S"),
    })

    # ── ROW 1: image + result ──────────────────
    col_img, col_res = st.columns([1.15, 0.85], gap="large")

    with col_img:
        b64 = img_to_b64(img_np)
        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-title">Input frame — {uploaded_file.name}</div>'
            f'<div class="scan-wrap">'
            f'<img src="data:image/png;base64,{b64}" alt="steel surface"/>'
            f'<div class="scan-line"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_res:
        gauge_svg = svg_gauge(confidence, chip_color)
        gauge_block = (
            f'<div class="gauge-wrap">{gauge_svg}'
            f'<div class="gauge-num">{pct}'
            f'<span style="font-size:1.1rem;color:#3a4260;font-weight:300">%</span>'
            f'</div>'
            f'<div class="gauge-lbl">Confidence</div>'
            f'</div>'
            f'<div class="prog">'
            f'<div class="prog-bar" style="width:{pct}%;background:{chip_color}"></div>'
            f'</div>'
        ) if show_conf else ""

        risk_html = risk_badge_html(risk_str, risk_color, risk_bg, risk_border)

        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-title">Detection Result</div>'

            f'<div class="defect-chip" style="background:{risk_bg};border:1px solid {risk_border}">'
            f'<span style="width:9px;height:9px;background:{chip_color};'
            f'border-radius:50%;display:inline-block"></span>'
            f'<span style="color:{chip_color}">{CLASS_DISPLAY[pred_label]}</span>'
            f'</div>'
            f'{risk_html}'

            f'<div class="defect-desc">{CLASS_DESC[pred_label]}</div>'

            f'{gauge_block}'

            f'<div class="tiles">'
            f'<div class="tile"><div class="tile-lbl">Confidence</div>'
            f'<div class="tile-val" style="color:{chip_color}">{pct}%</div></div>'
            f'<div class="tile"><div class="tile-lbl">Risk level</div>'
            f'<div class="tile-val" style="color:{risk_color}">{risk_str}</div></div>'
            f'<div class="tile"><div class="tile-lbl">Defect class</div>'
            f'<div class="tile-val" style="font-size:.75rem">{pred_idx + 1}/6</div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── ROW 2: probability distribution ────────
    st.markdown('<div class="sec-lbl">Defect Probability Distribution</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel">'
        f'<div class="panel-title">All class scores — top prediction highlighted</div>'
        f'{prob_bar_html(probs, pred_idx)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── ROW 3: heatmaps ────────────────────────
    if show_heatmap:
        st.markdown('<div class="sec-lbl">Defect Focus Maps</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3, gap="medium")

        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # CLAHE
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Canny + Inferno
        edges    = cv2.Canny(gray, 80, 180)
        inferno  = cv2.applyColorMap(edges, cv2.COLORMAP_INFERNO)
        ov_inf   = cv2.addWeighted(
            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, inferno, 0.4, 0)

        # Laplacian + Jet
        lap      = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        jet      = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
        ov_jet   = cv2.addWeighted(
            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.55, jet, 0.45, 0)

        with h1:
            st.markdown(
                '<div class="panel"><div class="panel-title">CLAHE enhanced</div>',
                unsafe_allow_html=True)
            st.image(enhanced, use_container_width=True, clamp=True)
            st.markdown(
                '<div class="legend"><span>Dark</span>'
                '<div class="lgd-grey"></div><span>Bright</span></div></div>',
                unsafe_allow_html=True)

        with h2:
            st.markdown(
                '<div class="panel"><div class="panel-title">Canny edge + Inferno</div>',
                unsafe_allow_html=True)
            st.image(cv2.cvtColor(ov_inf, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(
                '<div class="legend"><span>Low</span>'
                '<div class="lgd-inferno"></div><span>High</span></div></div>',
                unsafe_allow_html=True)

        with h3:
            st.markdown(
                '<div class="panel"><div class="panel-title">Laplacian + Jet overlay</div>',
                unsafe_allow_html=True)
            st.image(cv2.cvtColor(ov_jet, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(
                '<div class="legend"><span>Low</span>'
                '<div class="lgd-jet"></div><span>High</span></div></div>',
                unsafe_allow_html=True)

    # ── DOWNLOAD ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    all_probs = "\n".join([
        f"   {CLASS_DISPLAY[c]:<20}: {probs[i]*100:.2f}%"
        for i, c in enumerate(CLASS_NAMES)
    ])
    report = "\n".join([
        "=" * 56,
        "   STEELGUARD AI — SURFACE DEFECT INSPECTION REPORT",
        "   Tata Steel · Quality Control Division",
        "=" * 56,
        f"   File         : {uploaded_file.name}",
        f"   Timestamp    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "   ── PREDICTION ──",
        f"   Defect Type  : {CLASS_DISPLAY[pred_label]}",
        f"   Confidence   : {confidence*100:.2f}%",
        f"   Risk Level   : {risk_str}",
        "",
        "   ── ALL CLASS PROBABILITIES ──",
        all_probs,
        "=" * 56,
    ])
    st.download_button(
        "📥 Download Inspection Report",
        data=report,
        file_name=f"steelguard_{uploaded_file.name.rsplit('.', 1)[0]}.txt",
        mime="text/plain",
    )

else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">🔬</div>'
        '<div class="empty-text">'
        'Upload a high-resolution steel plate image to begin surface defect inspection.<br>'
        '<span style="color:#3d4a60;font-family:\'IBM Plex Mono\',monospace;font-size:.78rem">'
        'JPG · JPEG · PNG &nbsp;·&nbsp; up to 200 MB</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )