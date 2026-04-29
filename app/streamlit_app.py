import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import datetime
import base64
import os
import sys
from io import BytesIO
import gdown

# ══════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
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
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]
CLASS_DISPLAY = {
    "crazing":         "Crazing",
    "inclusion":       "Inclusion",
    "patches":         "Patches",
    "pitted_surface":  "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches":       "Scratches",
}
RISK_MAP = {
    "scratches":        ("HIGH",   "#e74c3c", "#1a0606", "#4a1515"),
    "crazing":          ("HIGH",   "#e74c3c", "#1a0606", "#4a1515"),
    "rolled-in_scale":  ("MEDIUM", "#f39c12", "#1a1206", "#4a3010"),
    "pitted_surface":   ("MEDIUM", "#f39c12", "#1a1206", "#4a3010"),
    "inclusion":        ("LOW",    "#27ae60", "#061a0e", "#1a4028"),
    "patches":          ("LOW",    "#27ae60", "#061a0e", "#1a4028"),
}
CLASS_COLORS = {
    "crazing":         "#e74c3c",
    "inclusion":       "#3498db",
    "patches":         "#2ecc71",
    "pitted_surface":  "#f39c12",
    "rolled-in_scale": "#9b59b6",
    "scratches":       "#e67e22",
}
CLASS_DESC = {
    "crazing":         "Network of fine surface cracks forming a web-like pattern.",
    "inclusion":       "Foreign particles embedded during the rolling process.",
    "patches":         "Irregular discoloured or rough surface regions.",
    "pitted_surface":  "Small cavities or craters on the steel surface.",
    "rolled-in_scale": "Oxide scale pressed into the surface during hot-rolling.",
    "scratches":       "Linear marks caused by abrasion during handling or transport.",
}
RISK_SCORE = {"HIGH": 85, "MEDIUM": 50, "LOW": 20}

import os
import urllib.request

MODEL_PATH = "models/best_resnet50_crack_detector.h5"
os.makedirs("models", exist_ok=True)

print("START CHECK →", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading model from HuggingFace...")

    url = "https://huggingface.co/datasets/jai567/steelguard-model/resolve/main/best_resnet50_crack_detector.h5"
    
    urllib.request.urlretrieve(url, MODEL_PATH)

print("END CHECK →", os.path.exists(MODEL_PATH))

# 🔥 CRITICAL LINE
MODEL_SEARCH_PATHS = [MODEL_PATH]

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

section[data-testid="stSidebar"] { background:#0a0d14; border-right:1px solid #161c28; }
section[data-testid="stSidebar"] label { color:#606880 !important; font-size:0.8rem; }
section[data-testid="stSidebar"] .stSlider > div { padding:0; }

.hdr {
    display:flex;align-items:center;gap:16px;
    background:#0d1120;border:1px solid #1a2035;
    border-top:2px solid #c0392b;border-radius:0 0 12px 12px;
    padding:18px 24px;margin-bottom:22px;
}
.hdr-icon {
    width:46px;height:46px;background:linear-gradient(140deg,#9b1c1c,#e74c3c);
    border-radius:10px;font-size:22px;display:flex;align-items:center;
    justify-content:center;flex-shrink:0;box-shadow:0 4px 18px rgba(231,76,60,.3);
}
.hdr-name { font-size:1.45rem;font-weight:600;color:#edf0f7;letter-spacing:-.01em; }
.hdr-sub  { font-size:.7rem;color:#3d4560;font-family:'IBM Plex Mono',monospace;letter-spacing:.07em;margin-top:2px; }
.hdr-right { margin-left:auto;display:flex;align-items:center;gap:12px; }
.badge { padding:5px 13px;border-radius:20px;font-size:.67rem;font-family:'IBM Plex Mono',monospace;letter-spacing:.08em;font-weight:500; }
.badge-online { background:#061a0e;border:1px solid #1a4028;color:#27ae60; }
.badge-demo   { background:#1a1206;border:1px solid #4a3010;color:#f39c12; }
.badge-time   { background:#0d1120;border:1px solid #1a2035;color:#3d4560; }
.dot  { width:7px;height:7px;background:#27ae60;border-radius:50%;display:inline-block;margin-right:5px;animation:blink 2s infinite; }
.dotd { width:7px;height:7px;background:#f39c12;border-radius:50%;display:inline-block;margin-right:5px; }
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

.demo-notice {
    background:#0f0d06;border:1px solid #3a2c08;border-left:3px solid #f39c12;
    border-radius:10px;padding:14px 18px;margin-bottom:18px;font-size:.82rem;
}
.demo-notice strong{color:#f39c12;font-family:'IBM Plex Mono',monospace;}
.demo-notice p{color:#8a7a50;margin:4px 0 0;line-height:1.6;}
.demo-notice code{background:#1a1608;border:1px solid #3a2c08;padding:1px 6px;
    border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:#c9a84c;}

@keyframes scan{0%{top:0;opacity:.8}100%{top:100%;opacity:.05}}
.scan-wrap{position:relative;overflow:hidden;border-radius:8px;}
.scan-wrap img{display:block;width:100%;}
.scan-line{position:absolute;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#e74c3c88,#e74c3c,#e74c3c88,transparent);
    animation:scan 1.8s linear infinite;pointer-events:none;}

.panel{background:#0d1120;border:1px solid #1a2035;border-radius:12px;padding:20px;margin-bottom:16px;}
.panel-title{font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#3a4260;
    font-family:'IBM Plex Mono',monospace;border-bottom:1px solid #131826;padding-bottom:9px;margin-bottom:16px;}

.defect-chip{display:inline-flex;align-items:center;gap:8px;padding:7px 14px;
    border-radius:8px;font-size:.95rem;font-weight:600;margin-bottom:10px;}
.defect-desc{font-size:.8rem;color:#5c6680;line-height:1.6;margin-bottom:16px;
    padding-bottom:14px;border-bottom:1px solid #131826;}

.gauge-wrap{display:flex;flex-direction:column;align-items:center;margin:6px 0 4px;}
.gauge-num{font-size:2.5rem;font-weight:500;color:#edf0f7;font-family:'IBM Plex Mono',monospace;
    letter-spacing:-.04em;margin-top:-10px;}
.gauge-lbl{font-size:.62rem;letter-spacing:.12em;color:#3a4260;text-transform:uppercase;}

.risk-meter-wrap{margin:14px 0 16px;}
.risk-meter-lbl{font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;color:#3a4260;
    font-family:'IBM Plex Mono',monospace;margin-bottom:7px;display:flex;justify-content:space-between;}
.risk-track{background:#131826;border-radius:4px;height:10px;width:100%;overflow:hidden;}
.risk-fill{height:10px;border-radius:4px;}

.prog{background:#131826;border-radius:3px;height:4px;width:100%;overflow:hidden;margin:8px 0 16px;}
.prog-bar{height:4px;border-radius:3px;}

.tiles{display:flex;gap:10px;}
.tile{flex:1;background:#080b11;border:1px solid #161c28;border-radius:10px;padding:12px 14px;}
.tile-lbl{font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:#3a4260;
    font-family:'IBM Plex Mono',monospace;margin-bottom:5px;}
.tile-val{font-size:1.05rem;font-weight:600;color:#b0b8cc;font-family:'IBM Plex Mono',monospace;}

.prob-row{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.prob-name{font-size:.72rem;color:#8892a4;font-family:'IBM Plex Mono',monospace;
    width:130px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.prob-track{flex:1;background:#131826;border-radius:3px;height:8px;overflow:hidden;}
.prob-fill-bar{height:8px;border-radius:3px;}
.prob-pct{font-size:.7rem;color:#3a4260;font-family:'IBM Plex Mono',monospace;
    width:44px;text-align:right;flex-shrink:0;}
.prob-row.active .prob-name{color:#edf0f7;font-weight:500;}
.prob-row.active .prob-pct{color:#edf0f7;}

.map-footer{display:flex;align-items:center;gap:8px;margin-top:10px;
    font-size:.65rem;font-family:'IBM Plex Mono',monospace;color:#3a4260;}
.map-footer span{flex-shrink:0;}
.map-footer .bar{flex:1;height:6px;border-radius:3px;}

.hist-row{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;
    border:1px solid #131826;background:#080b11;margin-bottom:5px;}
.hdot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.hname{color:#8892a4;flex:1;font-family:'IBM Plex Mono',monospace;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap;font-size:.67rem;}
.hlabel{font-family:'IBM Plex Mono',monospace;font-size:.67rem;color:#5c6680;}
.hconf{color:#3a4260;font-family:'IBM Plex Mono',monospace;min-width:44px;text-align:right;font-size:.67rem;}
.htime{color:#222838;font-family:'IBM Plex Mono',monospace;font-size:.6rem;}

.sec-lbl{font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:#3a4260;
    font-family:'IBM Plex Mono',monospace;margin:18px 0 10px;}
[data-testid="stFileUploader"]{border:1.5px dashed #1a2035!important;
    border-radius:10px!important;background:#0a0d14!important;}
.stDownloadButton>button{background:#0d1120!important;border:1px solid #1a2035!important;
    color:#e74c3c!important;font-family:'IBM Plex Mono',monospace!important;
    font-size:.72rem!important;letter-spacing:.06em!important;
    border-radius:8px!important;padding:8px 20px!important;transition:all .2s;}
.stDownloadButton>button:hover{background:#e74c3c!important;color:#fff!important;border-color:#e74c3c!important;}
.empty-state{background:#0a0d14;border:1.5px dashed #161c28;border-radius:12px;padding:60px 32px;text-align:center;}
.empty-icon{font-size:2.4rem;margin-bottom:12px;}
.empty-text{font-size:.85rem;line-height:1.8;color:#3a4260;}
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
    return (
        f'<svg width="172" height="92" viewBox="0 0 172 92">'
        f'<path d="M20,86 A66,66 0 0,1 152,86" fill="none" stroke="#131826" stroke-width="9" stroke-linecap="round"/>'
        f'<path d="M20,86 A66,66 0 0,1 152,86" fill="none" stroke="{color}" stroke-width="9" stroke-linecap="round"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}"/>'
        f'</svg>'
    )

def img_to_b64(img_rgb: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def risk_badge_html(risk: str, color: str, bg: str, border: str) -> str:
    return (
        f'<span style="background:{bg};border:1px solid {border};color:{color};'
        f'padding:4px 12px;border-radius:20px;font-size:.68rem;'
        f'font-family:\'IBM Plex Mono\',monospace;letter-spacing:.08em;font-weight:600">{risk}</span>'
    )

def risk_meter_html(risk: str, risk_color: str) -> str:
    score = RISK_SCORE[risk]
    return (
        f'<div class="risk-meter-wrap">'
        f'<div class="risk-meter-lbl"><span>Risk Score</span>'
        f'<span style="color:{risk_color}">{score}/100</span></div>'
        f'<div class="risk-track">'
        f'<div class="risk-fill" style="width:{score}%;background:{risk_color}"></div>'
        f'</div></div>'
    )

def legend_bar(label_left: str, label_right: str, gradient: str) -> str:
    return (
        f'<div class="map-footer">'
        f'<span>{label_left}</span>'
        f'<div class="bar" style="background:{gradient}"></div>'
        f'<span>{label_right}</span>'
        f'</div>'
    )

def prob_bar_html(probs: np.ndarray, pred_idx: int, threshold: float) -> str:
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        p      = float(probs[i])
        col    = CLASS_COLORS[cls]
        active = "active" if i == pred_idx else ""
        opacity = "1" if p >= threshold else "0.3"
        rows.append(
            f'<div class="prob-row {active}">'
            f'<div class="prob-name">{CLASS_DISPLAY[cls]}</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill-bar" style="width:{p*100:.1f}%;background:{col};opacity:{opacity}"></div>'
            f'</div>'
            f'<div class="prob-pct">{p*100:.1f}%</div>'
            f'</div>'
        )
    return "".join(rows)

def demo_predict(img_np: np.ndarray) -> np.ndarray:
    """Texture-based heuristic prediction used in demo mode."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean = float(gray.mean())
    rng  = np.random.default_rng(seed=int(var * 1000) % 99999)
    raw  = rng.dirichlet(np.ones(6) * 0.5)
    if var > 400:    raw[5] += 0.4
    elif var > 150:  raw[0] += 0.35
    elif mean < 80:  raw[3] += 0.35
    elif mean > 180: raw[2] += 0.35
    else:            raw[1] += 0.35
    return (raw / raw.sum()).astype(np.float32)

def normalize_probs(raw: np.ndarray) -> np.ndarray:
    """Normalize model output to 6-class probability vector."""
    if raw.ndim == 0:
        raw = np.array([float(raw)])
    raw = raw.flatten()
    if raw.shape[0] == 1:
        p = float(raw[0])
        out = np.zeros(6, dtype=np.float32)
        if p > 0.5: out[5], out[2] = p, 1 - p
        else:       out[2], out[5] = 1 - p, p
        return out
    if raw.shape[0] == 2:
        p = float(raw[1])
        out = np.zeros(6, dtype=np.float32)
        if p > 0.5: out[5], out[2] = p, 1 - p
        else:       out[2], out[5] = 1 - p, p
        return out
    if raw.shape[0] == 6:
        return raw.astype(np.float32)
    # Unknown shape — return uniform
    return np.ones(6, dtype=np.float32) / 6

# ══════════════════════════════════════════════
#  MODEL LOADING  — completely safe, never raises
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_safe():
    """
    Try to load the Keras model from any known path.
    Returns (model_object, path_string) or (None, None).
    Never raises — all exceptions are caught.
    """
    # Attempt TF / Keras imports
    load_fn = None
    try:
        from tensorflow.keras.models import load_model as _lm
        load_fn = _lm
    except ImportError:
        pass

    if load_fn is None:
        try:
            from keras.models import load_model as _lm
            load_fn = _lm
        except ImportError:
            print("[SteelGuard] Neither tensorflow.keras nor keras found.", file=sys.stderr)
            return None, None

    for p in MODEL_SEARCH_PATHS:
        if os.path.exists(p):
            print(f"[SteelGuard] Trying model at: {p}", file=sys.stderr)
            try:
                mdl = load_fn(p)
                print(f"[SteelGuard] Model loaded successfully from: {p}", file=sys.stderr)
                return mdl, p
            except Exception as e:
                print(f"[SteelGuard] Failed to load {p}: {e}", file=sys.stderr)
                continue

    print("[SteelGuard] No model file found. Running in DEMO mode.", file=sys.stderr)
    print(f"[SteelGuard] Searched: {MODEL_SEARCH_PATHS}", file=sys.stderr)
    return None, None


# ── Load model (safe) ───────────────────────
try:
    model, model_path = load_model_safe()
except Exception as _e:
    print(f"[SteelGuard] Unexpected error in load_model_safe: {_e}", file=sys.stderr)
    model, model_path = None, None

# These are ALWAYS defined regardless of model load outcome
MODEL_LOADED: bool = model is not None
DEMO_MODE:    bool = not MODEL_LOADED

# ══════════════════════════════════════════════
#  GRAD-CAM  — safe, always returns a heatmap
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_grad_model(_model):
    """Build a Grad-CAM sub-model. Returns None on any failure."""
    if _model is None:
        return None
    try:
        import tensorflow as tf

        conv_types = (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.SeparableConv2D,
        )

        def all_layers(m):
            for layer in m.layers:
                if hasattr(layer, "layers"):
                    yield from all_layers(layer)
                else:
                    yield layer

        last_conv = None
        for layer in all_layers(_model):
            if isinstance(layer, conv_types):
                last_conv = layer

        if last_conv is None:
            return None

        grad_m = tf.keras.models.Model(
            inputs=_model.inputs,
            outputs=[last_conv.output, _model.output],
        )
        return grad_m
    except Exception as e:
        print(f"[SteelGuard] build_grad_model failed: {e}", file=sys.stderr)
        return None


try:
    grad_model = build_grad_model(model)
except Exception as _e:
    print(f"[SteelGuard] grad_model build error: {_e}", file=sys.stderr)
    grad_model = None


def get_gradcam(img_array: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    Returns a float32 heatmap [0,1].  Always succeeds.
    Uses real Grad-CAM when grad_model is available, else texture pseudo-heatmap.
    """
    if grad_model is not None:
        try:
            import tensorflow as tf
            img_tensor = tf.cast(img_array, tf.float32)
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_tensor)
                tape.watch(conv_out)
                class_score = preds[:, pred_idx]
            grads   = tape.gradient(class_score, conv_out)
            pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (heatmap.max() + 1e-8)
            return heatmap.astype(np.float32)
        except Exception as e:
            print(f"[SteelGuard] Grad-CAM failed: {e}", file=sys.stderr)

    # Pseudo Grad-CAM — texture-based, always works
    orig = img_array[0]
    gray = cv2.cvtColor((orig * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap  = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap  = cv2.GaussianBlur(lap, (15, 15), 0)
    lap  = cv2.GaussianBlur(lap, (15, 15), 0)
    lap /= (lap.max() + 1e-8)
    return lap.astype(np.float32)


def apply_gradcam_overlay(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w    = img_np.shape[:2]
    hm      = cv2.resize(heatmap, (w, h))
    hm_u    = np.uint8(255 * hm)
    colored = cv2.applyColorMap(hm_u, cv2.COLORMAP_JET)
    bgr     = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr, 1 - alpha, colored, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
now_str = datetime.datetime.now().strftime("%H:%M:%S")

st.sidebar.markdown(
    '<div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;'
    'color:#3a4260;font-family:\'IBM Plex Mono\',monospace;padding:6px 0 14px">'
    'Model Controls</div>', unsafe_allow_html=True)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.05)
gradcam_alpha        = st.sidebar.slider("Grad-CAM Overlay Strength", 0.1, 0.9, 0.45, 0.05)

st.sidebar.markdown("<hr style='border-color:#161c28;margin:14px 0'>", unsafe_allow_html=True)

show_conf    = st.sidebar.checkbox("Show Confidence Gauge",  True)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap",  True)
show_heatmap = st.sidebar.checkbox("Show CV Focus Maps",      True)
show_probs   = st.sidebar.checkbox("Show Probability Bars",   True)
show_hist    = st.sidebar.checkbox("Show Inspection History", True)

st.sidebar.markdown("<hr style='border-color:#161c28;margin:14px 0'>", unsafe_allow_html=True)

model_path_lbl = os.path.basename(model_path) if model_path else "not found"
model_status   = "✓ Loaded"         if MODEL_LOADED      else "✗ Not found"
gradcam_status = "✓ Real Grad-CAM"  if grad_model is not None else "~ Pseudo (texture)"

st.sidebar.markdown(
    f'<div style="font-size:.62rem;color:#28304a;font-family:\'IBM Plex Mono\',monospace;line-height:2.1">'
    f'MODEL &nbsp;&nbsp;&nbsp;ResNet-50 v2<br>'
    f'TASK &nbsp;&nbsp;&nbsp;&nbsp;6-Class Detection<br>'
    f'INPUT &nbsp;&nbsp;&nbsp;224 × 224 × 3<br>'
    f'FILE &nbsp;&nbsp;&nbsp;&nbsp;{model_path_lbl}<br>'
    f'STATUS &nbsp;&nbsp;{model_status}<br>'
    f'GRAD-CAM &nbsp;{gradcam_status}<br>'
    f'CLOCK &nbsp;&nbsp;&nbsp;{now_str}'
    f'</div>', unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border-color:#161c28;margin:14px 0'>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;'
    'color:#3a4260;font-family:\'IBM Plex Mono\',monospace;margin-bottom:10px">'
    'Defect Classes</div>', unsafe_allow_html=True)

for cls in CLASS_NAMES:
    risk, col, bg, border = RISK_MAP[cls]
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        f'<div style="width:8px;height:8px;background:{col};border-radius:50%;flex-shrink:0"></div>'
        f'<span style="font-size:.7rem;color:#8892a4;font-family:\'IBM Plex Mono\',monospace;flex:1">'
        f'{CLASS_DISPLAY[cls]}</span>'
        f'<span style="font-size:.6rem;color:{col};font-family:\'IBM Plex Mono\',monospace">{risk}</span>'
        f'</div>', unsafe_allow_html=True)

if show_hist and st.session_state.history:
    st.sidebar.markdown("<hr style='border-color:#161c28;margin:14px 0'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div style="font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;'
        'color:#3a4260;font-family:\'IBM Plex Mono\',monospace;margin-bottom:10px">'
        'Recent Inspections</div>', unsafe_allow_html=True)
    for h in reversed(st.session_state.history[-8:]):
        col = CLASS_COLORS.get(h["label"], "#888")
        st.sidebar.markdown(
            f'<div class="hist-row">'
            f'<div class="hdot" style="background:{col}"></div>'
            f'<div class="hname">{h["name"]}</div>'
            f'<div class="hlabel">{CLASS_DISPLAY.get(h["label"], h["label"])}</div>'
            f'<div class="hconf">{h["conf"]*100:.1f}%</div>'
            f'<div class="htime">{h["time"]}</div>'
            f'</div>', unsafe_allow_html=True)
    if st.sidebar.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
now_full   = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
badge_html = (
    '<div class="badge badge-online"><span class="dot"></span>MODEL LOADED</div>'
    if MODEL_LOADED else
    '<div class="badge badge-demo"><span class="dotd"></span>DEMO MODE</div>'
)
st.markdown(
    f'<div class="hdr">'
    f'<div class="hdr-icon">🛡️</div>'
    f'<div>'
    f'<div class="hdr-name">SteelGuard AI</div>'
    f'<div class="hdr-sub">SURFACE DEFECT DETECTION + XAI · TATA STEEL QC DIVISION</div>'
    f'</div>'
    f'<div class="hdr-right">'
    f'<div class="badge badge-time">{now_full}</div>'
    f'{badge_html}'
    f'</div>'
    f'</div>', unsafe_allow_html=True)

if DEMO_MODE:
    st.markdown(
        '<div class="demo-notice">'
        '<strong>⚠ DEMO MODE — No model file found</strong>'
        '<p>Place your model at <code>models/best_resnet50_crack_detector.h5</code> '
        'in the repo root (or <code>models/best_model.keras</code>). '
        'Predictions currently use image-texture heuristics. '
        'Check the deployment logs for the exact searched paths.</p>'
        '</div>', unsafe_allow_html=True)

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
#  INFERENCE + DISPLAY
# ══════════════════════════════════════════════
if uploaded_file is not None:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_res   = cv2.resize(img_np, (224, 224)) / 255.0
    img_input = np.expand_dims(img_res, axis=0).astype(np.float32)

    with st.spinner("Running inference…"):
        time.sleep(0.3)
        try:
            if MODEL_LOADED and model is not None:
                raw   = model.predict(img_input, verbose=0)[0]
                probs = normalize_probs(raw)
            else:
                probs = demo_predict(img_np)
        except Exception as _inf_err:
            print(f"[SteelGuard] Inference error: {_inf_err}", file=sys.stderr)
            probs = demo_predict(img_np)

        pred_idx   = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        pct        = int(confidence * 100)
        flagged    = confidence < confidence_threshold

        risk_str, risk_color, risk_bg, risk_border = RISK_MAP[pred_label]
        chip_color = CLASS_COLORS[pred_label]

        heatmap = get_gradcam(img_input, pred_idx) if show_gradcam else None

    st.session_state.history.append({
        "name":  uploaded_file.name,
        "label": pred_label,
        "conf":  confidence,
        "time":  datetime.datetime.now().strftime("%H:%M:%S"),
    })

    # ── ROW 1: Image | Result ──────────────────
    col_img, col_res = st.columns([1.15, 0.85], gap="large")

    with col_img:
        b64      = img_to_b64(img_np)
        demo_tag = ' <span style="color:#f39c12;font-size:.58rem">[DEMO]</span>' if DEMO_MODE else ""
        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-title">Input frame — {uploaded_file.name}{demo_tag}</div>'
            f'<div class="scan-wrap">'
            f'<img src="data:image/png;base64,{b64}" alt="steel surface"/>'
            f'<div class="scan-line"></div>'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

    with col_res:
        gauge_svg   = svg_gauge(confidence, chip_color)
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

        inconclusive_html = (
            f'<div style="background:#0d0f17;border:1px solid #2a2e40;border-left:3px solid #3498db;'
            f'border-radius:8px;padding:10px 14px;margin:10px 0;font-size:.78rem;color:#6878a8">'
            f'⚠ Confidence {pct}% below threshold ({int(confidence_threshold*100)}%)'
            f' — result may be inconclusive.</div>'
        ) if flagged else ""

        gradcam_tag = "" if grad_model is not None else " (pseudo)"

        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-title">Detection Result · Grad-CAM{gradcam_tag}</div>'
            f'<div class="defect-chip" style="background:{risk_bg};border:1px solid {risk_border}">'
            f'<span style="width:9px;height:9px;background:{chip_color};'
            f'border-radius:50%;display:inline-block"></span>'
            f'<span style="color:{chip_color}">{CLASS_DISPLAY[pred_label]}</span>'
            f'</div> {risk_badge_html(risk_str, risk_color, risk_bg, risk_border)}'
            f'{inconclusive_html}'
            f'<div class="defect-desc">{CLASS_DESC[pred_label]}</div>'
            f'{gauge_block}'
            f'{risk_meter_html(risk_str, risk_color)}'
            f'<div class="tiles">'
            f'<div class="tile"><div class="tile-lbl">Confidence</div>'
            f'<div class="tile-val" style="color:{chip_color}">{pct}%</div></div>'
            f'<div class="tile"><div class="tile-lbl">Risk level</div>'
            f'<div class="tile-val" style="color:{risk_color}">{risk_str}</div></div>'
            f'<div class="tile"><div class="tile-lbl">Risk score</div>'
            f'<div class="tile-val" style="color:{risk_color}">{RISK_SCORE[risk_str]}/100</div></div>'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

    # ── ROW 2: Visualisation maps ──────────────
    if show_gradcam or show_heatmap:
        st.markdown('<div class="sec-lbl">Defect Visualisation Maps</div>', unsafe_allow_html=True)

        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges    = cv2.Canny(gray, 80, 180)
        inferno  = cv2.applyColorMap(edges, cv2.COLORMAP_INFERNO)
        ov_inf   = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, inferno, 0.4, 0)
        lap      = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        jet_map  = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
        ov_jet   = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.55, jet_map, 0.45, 0)

        panels = []
        if show_gradcam:  panels.append("gradcam")
        if show_heatmap:  panels += ["clahe", "inferno", "jet"]

        cols = st.columns(len(panels), gap="medium")

        GRAD_TAG  = "Grad-CAM (XAI)" if grad_model is not None else "Pseudo Grad-CAM (texture)"
        G_GREY    = "linear-gradient(90deg,#000000,#ffffff)"
        G_INFERNO = "linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#f0f921)"
        G_JET     = "linear-gradient(90deg,#000080,#00ffff,#ffff00,#ff0000)"

        for col_widget, panel in zip(cols, panels):
            with col_widget:
                if panel == "gradcam" and heatmap is not None:
                    st.markdown(
                        f'<div class="panel"><div class="panel-title">{GRAD_TAG}</div>',
                        unsafe_allow_html=True)
                    overlay = apply_gradcam_overlay(img_np, heatmap, gradcam_alpha)
                    st.image(overlay, use_container_width=True)
                    st.markdown(legend_bar("Cold", "Hot", G_JET) + "</div>", unsafe_allow_html=True)

                elif panel == "clahe":
                    st.markdown(
                        '<div class="panel"><div class="panel-title">CLAHE Enhanced</div>',
                        unsafe_allow_html=True)
                    st.image(enhanced, use_container_width=True, clamp=True)
                    st.markdown(legend_bar("Dark", "Bright", G_GREY) + "</div>", unsafe_allow_html=True)

                elif panel == "inferno":
                    st.markdown(
                        '<div class="panel"><div class="panel-title">Canny + Inferno</div>',
                        unsafe_allow_html=True)
                    st.image(cv2.cvtColor(ov_inf, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown(legend_bar("Low", "High", G_INFERNO) + "</div>", unsafe_allow_html=True)

                elif panel == "jet":
                    st.markdown(
                        '<div class="panel"><div class="panel-title">Laplacian + Jet</div>',
                        unsafe_allow_html=True)
                    st.image(cv2.cvtColor(ov_jet, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown(legend_bar("Low", "High", G_JET) + "</div>", unsafe_allow_html=True)

    # ── ROW 3: Probability bars ────────────────
    if show_probs:
        st.markdown('<div class="sec-lbl">Defect Probability Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-title">All class scores — bars below threshold '
            f'({int(confidence_threshold*100)}%) dimmed</div>'
            f'{prob_bar_html(probs, pred_idx, confidence_threshold)}'
            f'</div>', unsafe_allow_html=True)

    # ── DOWNLOAD ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    all_probs_txt = "\n".join([
        f"   {CLASS_DISPLAY[c]:<20}: {probs[i]*100:.2f}%"
        for i, c in enumerate(CLASS_NAMES)
    ])
    inference_note = ("DEMO MODE (texture heuristic)" if DEMO_MODE
                      else f"AI model: {os.path.basename(model_path)}")
    grad_note = "Real Grad-CAM" if grad_model is not None else "Pseudo Grad-CAM (texture fallback)"
    report = "\n".join([
        "=" * 58,
        "   STEELGUARD AI — SURFACE DEFECT INSPECTION REPORT",
        "   Tata Steel · Quality Control Division",
        "=" * 58,
        f"   File          : {uploaded_file.name}",
        f"   Timestamp     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"   Inference     : {inference_note}",
        f"   Grad-CAM      : {grad_note}",
        f"   Threshold     : {int(confidence_threshold*100)}%",
        "",
        "   ── PREDICTION ──────────────────────────────────",
        f"   Defect Type   : {CLASS_DISPLAY[pred_label]}",
        f"   Confidence    : {confidence*100:.2f}%",
        f"   Risk Level    : {risk_str}",
        f"   Risk Score    : {RISK_SCORE[risk_str]}/100",
        f"   Inconclusive  : {'Yes' if flagged else 'No'}",
        f"   Description   : {CLASS_DESC[pred_label]}",
        "",
        "   ── ALL CLASS PROBABILITIES ─────────────────────",
        all_probs_txt,
        "=" * 58,
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
        '</div>', unsafe_allow_html=True)
