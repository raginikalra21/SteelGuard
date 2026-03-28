import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import time
import datetime
import base64
from io import BytesIO

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="SteelGuard AI · Tata Steel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #080b11; }
.block-container { padding: 1.2rem 2rem 3rem; max-width: 1440px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0a0d14;
    border-right: 1px solid #161c28;
}
section[data-testid="stSidebar"] label { color: #606880 !important; font-size: 0.8rem; }

/* ── HEADER ── */
.hdr {
    display: flex; align-items: center; gap: 16px;
    background: #0d1120;
    border: 1px solid #1a2035;
    border-top: 2px solid #c0392b;
    border-radius: 0 0 12px 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
}
.hdr-icon {
    width: 44px; height: 44px;
    background: linear-gradient(140deg,#9b1c1c,#e74c3c);
    border-radius: 10px;
    display:flex;align-items:center;justify-content:center;
    font-size:20px;flex-shrink:0;
    box-shadow: 0 4px 16px rgba(231,76,60,0.3);
}
.hdr-name  { font-size:1.4rem;font-weight:600;color:#edf0f7;letter-spacing:-.01em; }
.hdr-sub   { font-size:.72rem;color:#3d4560;font-family:'IBM Plex Mono',monospace;letter-spacing:.06em;margin-top:2px; }
.hdr-right { margin-left:auto;display:flex;align-items:center;gap:12px; }
.badge { padding:5px 13px;border-radius:20px;font-size:.68rem;font-family:'IBM Plex Mono',monospace;letter-spacing:.08em;font-weight:500; }
.badge-online { background:#061a0e;border:1px solid #1a4028;color:#27ae60; }
.badge-time   { background:#0d1120;border:1px solid #1a2035;color:#3d4560; }
.dot { width:7px;height:7px;background:#27ae60;border-radius:50%;display:inline-block;margin-right:5px;animation:blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:.3} }

/* ── SCAN ANIMATION ── */
@keyframes scan { 0%{top:0;opacity:.8}100%{top:100%;opacity:.05} }
.scan-wrap { position:relative;overflow:hidden;border-radius:8px; }
.scan-wrap img { display:block;width:100%; }
.scan-line {
    position:absolute;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#e74c3c88,#e74c3c,#e74c3c88,transparent);
    animation:scan 1.8s linear infinite;pointer-events:none;
}

/* ── PANELS ── */
.panel {
    background:#0d1120;border:1px solid #1a2035;
    border-radius:12px;padding:20px;margin-bottom:16px;
}
.panel-title {
    font-size:.63rem;letter-spacing:.15em;text-transform:uppercase;
    color:#3a4260;font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid #131826;padding-bottom:9px;margin-bottom:15px;
}

/* ── VERDICT ── */
.verdict-wrap { display:flex;align-items:center;gap:12px;margin-bottom:14px; }
.verdict-icon { width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:17px; }
.icon-safe  { background:#061a0e;border:1px solid #1a4028; }
.icon-crack { background:#1a0606;border:1px solid #4a1515; }
.verdict-text { font-size:1.5rem;font-weight:600;line-height:1.1; }
.v-safe  { color:#27ae60; }
.v-crack { color:#e74c3c; }

/* ── GAUGE ── */
.gauge-wrap { display:flex;flex-direction:column;align-items:center;margin:6px 0 2px; }
.gauge-num {
    font-size:2.5rem;font-weight:500;
    font-family:'IBM Plex Mono',monospace;
    letter-spacing:-.04em;color:#edf0f7;margin-top:-10px;
}
.gauge-lbl { font-size:.62rem;letter-spacing:.12em;color:#3a4260;text-transform:uppercase;margin-top:1px; }

/* ── PROGRESS ── */
.prog { background:#131826;border-radius:3px;height:4px;width:100%;overflow:hidden;margin:8px 0 16px; }
.prog-bar { height:4px;border-radius:3px; }
.bar-safe  { background:linear-gradient(90deg,#1a6640,#27ae60); }
.bar-crack { background:linear-gradient(90deg,#7a1010,#e74c3c); }

/* ── TILES ── */
.tiles { display:flex;gap:10px; }
.tile { flex:1;background:#080b11;border:1px solid #161c28;border-radius:10px;padding:12px 14px; }
.tile-lbl { font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:#3a4260;font-family:'IBM Plex Mono',monospace;margin-bottom:5px; }
.tile-val { font-size:1.05rem;font-weight:600;color:#b0b8cc;font-family:'IBM Plex Mono',monospace; }
.val-hi { color:#e74c3c; }
.val-lo { color:#27ae60; }

/* ── HEATMAP LEGEND ── */
.legend { display:flex;align-items:center;gap:6px;margin-top:8px;font-size:.6rem;font-family:'IBM Plex Mono',monospace;color:#3a4260; }
.lgd-grey    { flex:1;height:5px;border-radius:3px;background:linear-gradient(90deg,#000,#fff); }
.lgd-inferno { flex:1;height:5px;border-radius:3px;background:linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#f0f921); }
.lgd-jet     { flex:1;height:5px;border-radius:3px;background:linear-gradient(90deg,#000080,#00ffff,#ffff00,#ff0000); }

/* ── HISTORY ROWS ── */
.hist-row { display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;border:1px solid #131826;background:#080b11;margin-bottom:5px;font-size:.72rem; }
.hdot-s { width:6px;height:6px;background:#27ae60;border-radius:50%;flex-shrink:0; }
.hdot-c { width:6px;height:6px;background:#e74c3c;border-radius:50%;flex-shrink:0; }
.hname  { color:#8892a4;flex:1;font-family:'IBM Plex Mono',monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.68rem; }
.hconf  { color:#3a4260;font-family:'IBM Plex Mono',monospace;min-width:48px;text-align:right;font-size:.68rem; }
.htime  { color:#222838;font-family:'IBM Plex Mono',monospace;font-size:.6rem; }

/* ── MISC ── */
.sec-lbl { font-size:.63rem;letter-spacing:.15em;text-transform:uppercase;color:#3a4260;font-family:'IBM Plex Mono',monospace;margin:16px 0 10px; }
[data-testid="stFileUploader"] { border:1.5px dashed #1a2035 !important;border-radius:10px !important;background:#0a0d14 !important; }
.stDownloadButton > button {
    background:#0d1120 !important;border:1px solid #1a2035 !important;
    color:#e74c3c !important;font-family:'IBM Plex Mono',monospace !important;
    font-size:.72rem !important;letter-spacing:.06em !important;
    border-radius:8px !important;padding:8px 20px !important;transition:all .2s;
}
.stDownloadButton > button:hover { background:#e74c3c !important;color:#fff !important;border-color:#e74c3c !important; }
.empty-state { background:#0a0d14;border:1.5px dashed #161c28;border-radius:12px;padding:60px 32px;text-align:center; }
.empty-icon  { font-size:2.4rem;margin-bottom:12px; }
.empty-text  { font-size:.85rem;line-height:1.8;color:#3a4260; }
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:#080b11}::-webkit-scrollbar-thumb{background:#161c28;border-radius:4px}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_crack_model():
    return load_model("models/resnet50_crack_detector.h5")

model = load_crack_model()

# ---------- HELPERS ----------
def svg_gauge(pct: float, is_crack: bool) -> str:
    r = 66; cx = 86; cy = 86
    circ = 3.14159 * r
    dash = circ * pct
    color  = "#e74c3c" if is_crack else "#27ae60"
    track  = "#1a0606"  if is_crack else "#061a0e"
    return f"""<svg width="172" height="92" viewBox="0 0 172 92">
  <path d="M20,86 A66,66 0 0,1 152,86" fill="none" stroke="{track}" stroke-width="9" stroke-linecap="round"/>
  <path d="M20,86 A66,66 0 0,1 152,86" fill="none" stroke="{color}" stroke-width="9" stroke-linecap="round"
        stroke-dasharray="{dash:.1f} {circ:.1f}"/>
</svg>"""

def img_to_b64(img_rgb):
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- SIDEBAR ----------
now_str = datetime.datetime.now().strftime("%H:%M:%S")

st.sidebar.markdown("""<div style='font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;
color:#3a4260;font-family:"IBM Plex Mono",monospace;padding:4px 0 12px'>Model Controls</div>""",
unsafe_allow_html=True)

threshold    = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
show_conf    = st.sidebar.checkbox("Show Confidence Gauge", True)
show_heatmap = st.sidebar.checkbox("Show Heatmap Panels", True)
show_hist    = st.sidebar.checkbox("Show Inspection History", True)

st.sidebar.markdown("<hr style='border-color:#161c28;margin:16px 0'>", unsafe_allow_html=True)
st.sidebar.markdown(f"""<div style='font-size:.62rem;color:#28304a;font-family:"IBM Plex Mono",monospace;line-height:2.1'>
MODEL &nbsp;&nbsp;&nbsp;&nbsp;ResNet-50 v2<br>
BACKBONE &nbsp;ImageNet → NEU-DET<br>
INPUT &nbsp;&nbsp;&nbsp;&nbsp;224 × 224 × 3<br>
CLASSES &nbsp;&nbsp;Crack / No-Crack<br>
THRESHOLD {threshold:.2f}<br>
CLOCK &nbsp;&nbsp;&nbsp;&nbsp;{now_str}
</div>""", unsafe_allow_html=True)

if show_hist and st.session_state.history:
    st.sidebar.markdown("<hr style='border-color:#161c28;margin:16px 0'>", unsafe_allow_html=True)
    st.sidebar.markdown("""<div style='font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;
color:#3a4260;font-family:"IBM Plex Mono",monospace;margin-bottom:10px'>Recent Inspections</div>""",
    unsafe_allow_html=True)
    for h in reversed(st.session_state.history[-7:]):
        dcls = "hdot-c" if h["crack"] else "hdot-s"
        st.sidebar.markdown(f"""<div class="hist-row">
  <div class="{dcls}"></div>
  <div class="hname">{h['name']}</div>
  <div class="hconf">{h['conf']*100:.1f}%</div>
  <div class="htime">{h['time']}</div>
</div>""", unsafe_allow_html=True)
    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------- HEADER ----------
now_full = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(f"""<div class="hdr">
  <div class="hdr-icon">🛡️</div>
  <div>
    <div class="hdr-name">SteelGuard AI</div>
    <div class="hdr-sub">MICRO-CRACK DETECTION SYSTEM · TATA STEEL QC DIVISION</div>
  </div>
  <div class="hdr-right">
    <div class="badge badge-time">{now_full}</div>
    <div class="badge badge-online"><span class="dot"></span>SYSTEM ONLINE</div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------- UPLOAD ----------
st.markdown('<div class="sec-lbl">Surface Image Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop steel plate image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ---------- MAIN ----------
if uploaded_file is not None:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_resized = cv2.resize(img_np, (224, 224)) / 255.0
    img_input   = np.expand_dims(img_resized, axis=0)

    with st.spinner("Running inference…"):
        time.sleep(0.4)
        prediction = float(model.predict(img_input, verbose=0)[0][0])

    is_crack   = prediction < threshold
    pred_label = "Crack Detected" if is_crack else "No Crack"
    confidence = (1 - prediction) if is_crack else prediction
    risk_str   = "HIGH" if is_crack else "LOW"
    pct        = int(confidence * 100)

    # log history
    st.session_state.history.append({
        "name": uploaded_file.name,
        "crack": is_crack,
        "conf": confidence,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    })

    col_img, col_res = st.columns([1.15, 0.85], gap="large")

    # ── IMAGE PANEL ──
    with col_img:
        b64 = img_to_b64(img_np)
        st.markdown(f"""<div class="panel">
  <div class="panel-title">Input frame — {uploaded_file.name}</div>
  <div class="scan-wrap">
    <img src="data:image/png;base64,{b64}" alt="steel surface"/>
    <div class="scan-line"></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── RESULT PANEL ──
    with col_res:
        icon_cls = "icon-crack" if is_crack else "icon-safe"
        icon_ch  = "⚠️" if is_crack else "✅"
        v_cls    = "v-crack" if is_crack else "v-safe"
        bar_cls  = "bar-crack" if is_crack else "bar-safe"
        gauge    = svg_gauge(confidence, is_crack)
        rval_cls = "val-hi" if is_crack else "val-lo"

        gauge_block = f"""
  <div class="gauge-wrap">{gauge}
    <div class="gauge-num">{pct}<span style="font-size:1.1rem;color:#3a4260;font-weight:300">%</span></div>
    <div class="gauge-lbl">Confidence</div>
  </div>
  <div class="prog"><div class="prog-bar {bar_cls}" style="width:{pct}%"></div></div>
""" if show_conf else ""

        st.markdown(f"""<div class="panel">
  <div class="panel-title">Prediction Result</div>
  <div class="verdict-wrap">
    <div class="verdict-icon {icon_cls}">{icon_ch}</div>
    <div class="verdict-text {v_cls}">{pred_label}</div>
  </div>
  {gauge_block}
  <div class="tiles">
    <div class="tile">
      <div class="tile-lbl">Raw score</div>
      <div class="tile-val">{prediction:.4f}</div>
    </div>
    <div class="tile">
      <div class="tile-lbl">Threshold</div>
      <div class="tile-val">{threshold:.2f}</div>
    </div>
    <div class="tile">
      <div class="tile-lbl">Risk level</div>
      <div class="tile-val {rval_cls}">{risk_str}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── HEATMAP SECTION ──
    if show_heatmap:
        st.markdown('<div class="sec-lbl">Defect Focus Maps</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3, gap="medium")

        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # A: CLAHE
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # B: Canny + Inferno
        edges    = cv2.Canny(gray, 80, 180)
        inferno  = cv2.applyColorMap(edges, cv2.COLORMAP_INFERNO)
        ov_inf   = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, inferno, 0.4, 0)

        # C: Laplacian + Jet
        lap      = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        jet      = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
        ov_jet   = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.55, jet, 0.45, 0)

        with h1:
            st.markdown('<div class="panel"><div class="panel-title">CLAHE enhanced</div>', unsafe_allow_html=True)
            st.image(enhanced, use_container_width=True, clamp=True)
            st.markdown('<div class="legend"><span>Dark</span><div class="lgd-grey"></div><span>Bright</span></div></div>', unsafe_allow_html=True)

        with h2:
            st.markdown('<div class="panel"><div class="panel-title">Canny edge + Inferno</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(ov_inf, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('<div class="legend"><span>Low</span><div class="lgd-inferno"></div><span>High</span></div></div>', unsafe_allow_html=True)

        with h3:
            st.markdown('<div class="panel"><div class="panel-title">Laplacian + Jet overlay</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(ov_jet, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('<div class="legend"><span>Low</span><div class="lgd-jet"></div><span>High</span></div></div>', unsafe_allow_html=True)

    # ── DOWNLOAD ──
    st.markdown("<br>", unsafe_allow_html=True)
    report = "\n".join([
        "=" * 54,
        "   STEELGUARD AI — INSPECTION REPORT",
        "   Tata Steel · Quality Control Division",
        "=" * 54,
        f"   File        : {uploaded_file.name}",
        f"   Timestamp   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"   Verdict     : {pred_label}",
        f"   Confidence  : {confidence*100:.2f}%",
        f"   Raw score   : {prediction:.6f}",
        f"   Threshold   : {threshold:.2f}",
        f"   Risk level  : {risk_str}",
        "=" * 54,
    ])
    st.download_button(
        "📥 Download Inspection Report",
        data=report,
        file_name=f"steelguard_{uploaded_file.name.rsplit('.',1)[0]}.txt",
        mime="text/plain"
    )

else:
    st.markdown("""<div class="empty-state">
  <div class="empty-icon">🔬</div>
  <div class="empty-text">
    Upload a high-resolution steel plate image to begin micro-crack inspection.<br>
    <span style="color:#3d4a60;font-family:'IBM Plex Mono',monospace;font-size:.78rem">JPG · JPEG · PNG &nbsp;·&nbsp; up to 200 MB</span>
  </div>
</div>""", unsafe_allow_html=True)