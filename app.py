import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from torchvision import transforms, models
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Metal Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# ATTRACTIVE AI PRELOADER ANIMATION
# -----------------------------------------------------------------------------
components.html("""
<style>
#ai-page-preloader {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(circle at center, #0F172A 0%, #060913 100%);
    z-index: 999999999;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: opacity 0.7s cubic-bezier(0.4, 0, 0.2, 1), visibility 0.7s, transform 0.7s;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}
#ai-page-preloader.fade-out {
    opacity: 0;
    visibility: hidden;
    transform: scale(1.04);
    pointer-events: none;
}
.preloader-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 20px;
}
.hologram-scanner {
    position: relative;
    width: 140px;
    height: 140px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 28px;
}
.glow-orbit-outer {
    position: absolute;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    border: 3px solid transparent;
    border-top-color: #38BDF8;
    border-bottom-color: #818CF8;
    animation: rotateClockwise 2s linear infinite;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.45);
}
.glow-orbit-inner {
    position: absolute;
    width: 105px;
    height: 105px;
    border-radius: 50%;
    border: 2px dashed rgba(6, 182, 212, 0.7);
    border-left-color: #10B981;
    animation: rotateCounter 2.8s linear infinite;
}
.laser-grid-line {
    position: absolute;
    width: 100px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38BDF8, #FFFFFF, #38BDF8, transparent);
    box-shadow: 0 0 12px #38BDF8;
    animation: sweepLaser 1.8s ease-in-out infinite alternate;
}
.scanner-core-icon {
    font-size: 2.6rem;
    filter: drop-shadow(0 0 16px rgba(56, 189, 248, 0.8));
    animation: pulseCore 1.8s ease-in-out infinite;
}
.preloader-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.progress-wrapper {
    width: 280px;
    margin-top: 6px;
}
.progress-rail {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}
.progress-beam {
    width: 6%;
    height: 100%;
    background: linear-gradient(90deg, #06B6D4, #3B82F6, #10B981);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.7);
    border-radius: 12px;
    transition: width 0.18s ease-out;
}
.progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace, sans-serif;
}
.preloader-status {
    color: #94A3B8;
    font-size: 0.82rem;
}
.progress-pct {
    color: #38BDF8;
    font-weight: 700;
    font-size: 0.85rem;
}
.preloader-badge {
    margin-top: 16px;
    font-size: 0.72rem;
    color: #64748B;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
}
@keyframes rotateClockwise {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes rotateCounter {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(-360deg); }
}
@keyframes sweepLaser {
    0% { transform: translateY(-42px); opacity: 0.3; }
    50% { opacity: 1; }
    100% { transform: translateY(42px); opacity: 0.3; }
}
@keyframes pulseCore {
    0%, 100% { transform: scale(0.95); opacity: 0.85; }
    50% { transform: scale(1.1); opacity: 1; }
}
</style>

<script>
(function() {
    const parentDoc = window.parent.document;
    if (parentDoc.getElementById('ai-page-preloader')) return;

    const preloader = parentDoc.createElement('div');
    preloader.id = 'ai-page-preloader';
    preloader.innerHTML = `
        <div class="preloader-card">
            <div class="hologram-scanner">
                <div class="glow-orbit-outer"></div>
                <div class="glow-orbit-inner"></div>
                <div class="laser-grid-line"></div>
                <div class="scanner-core-icon">🔬</div>
            </div>
            <div class="preloader-title">METAL DEFECT AI</div>
            <div class="progress-wrapper">
                <div class="progress-rail">
                    <div class="progress-beam" id="progress-fill-bar"></div>
                </div>
                <div class="progress-info">
                    <span class="preloader-status" id="preloader-status-text">Initializing Vision Engine...</span>
                    <span class="progress-pct" id="progress-pct-val">6%</span>
                </div>
            </div>
            <div class="preloader-badge">Deep Learning • Surface Vision Engine</div>
        </div>
    `;

    const styleEl = parentDoc.createElement('style');
    styleEl.textContent = document.querySelector('style').textContent;
    parentDoc.head.appendChild(styleEl);
    parentDoc.body.appendChild(preloader);

    let progress = 6;
    const bar = preloader.querySelector('#progress-fill-bar');
    const pct = preloader.querySelector('#progress-pct-val');
    const statusText = preloader.querySelector('#preloader-status-text');

    // Smooth, realistic slow progressive crawl
    const progressTimer = setInterval(() => {
        if (progress < 90) {
            const step = Math.max(0.2, (90 - progress) * 0.035);
            progress += step;
            if (progress > 90) progress = 90;

            if (bar) bar.style.width = progress.toFixed(1) + '%';
            if (pct) pct.textContent = Math.floor(progress) + '%';

            if (statusText) {
                if (progress > 70) {
                    statusText.textContent = 'Preparing Project View...';
                } else if (progress > 45) {
                    statusText.textContent = 'Loading ResNet-18 Weights...';
                } else if (progress > 20) {
                    statusText.textContent = 'Connecting Vision Engine...';
                }
            }
        }
    }, 70);

    window.parent.dismissAIPagePreloader = function() {
        clearInterval(progressTimer);

        // Smoothly accelerate to 100% completion
        if (bar) {
            bar.style.transition = 'width 0.45s ease-out';
            bar.style.width = '100%';
        }
        if (pct) pct.textContent = '100%';
        if (statusText) statusText.textContent = 'Ready! Loading Original Project...';

        setTimeout(() => {
            if (preloader && !preloader.classList.contains('fade-out')) {
                preloader.classList.add('fade-out');
                setTimeout(() => {
                    if (preloader && preloader.parentNode) {
                        preloader.parentNode.removeChild(preloader);
                    }
                }, 750);
            }
        }, 500);
    };

    // Safety fallback so it never hangs indefinitely
    setTimeout(() => {
        if (typeof window.parent.dismissAIPagePreloader === 'function') {
            window.parent.dismissAIPagePreloader();
        }
    }, 15000);
})();
</script>
""", height=0, width=0)

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
IMG_SIZE = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "defect_classifier.pth")
CONF_THRESHOLD = 0.60
DEVICE = "cpu"

DEMO_SAMPLES = {
    "None (Upload my own)": None,
    "Example 1: Surface Defect (Pit/Scratch)": os.path.join(BASE_DIR, "samples", "defect_pit.jpg") if os.path.exists(os.path.join(BASE_DIR, "samples", "defect_pit.jpg")) else os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "0000f269f.jpg"),
    "Example 2: Surface Defect (Crack)": os.path.join(BASE_DIR, "samples", "defect_crack.jpg") if os.path.exists(os.path.join(BASE_DIR, "samples", "defect_crack.jpg")) else os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "0002cc93b.jpg"),
    "Example 3: Normal Surface (Clean Metal)": os.path.join(BASE_DIR, "samples", "normal_surface.jpg") if os.path.exists(os.path.join(BASE_DIR, "samples", "normal_surface.jpg")) else os.path.join(BASE_DIR, "models", "Non diffected with greyscale", "kos01", "Part2.jpg"),
}

# -----------------------------------------------------------------------------
# CLEAN & SIMPLE STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #E2E8F0 !important;
        margin-bottom: 24px;
        font-weight: 400;
    }
    .result-card-defect {
        background: rgba(239, 68, 68, 0.18);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 16px 20px;
        color: #FEE2E2 !important;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 16px 0;
    }
    .result-card-normal {
        background: rgba(34, 197, 94, 0.18);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 16px 20px;
        color: #DCFCE7 !important;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin: 16px 0;
    }
    .image-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL INFERENCE
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            return model
        except Exception:
            return None
    return None

model = load_model()

def predict(img_3ch):
    if model is None:
        return 0.0
    x = transform(img_3ch).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)
    return prob[0, 1].item()

def create_heatmap(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)
    return heatmap, dilated

def highlight_defects_with_regions(gray, heatmap):
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    regions = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({"X": x, "Y": y, "Width": w, "Height": h, "Area": int(area)})
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output, regions

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>🔍 Metal Surface Defect Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Easily upload or select an image to check if a metal surface is normal or defective.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GLOBAL DRAG & DROP ANYWHERE ON PAGE
# -----------------------------------------------------------------------------
components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;
    if (parentDoc.getElementById('global-drag-installed')) return;

    const marker = parentDoc.createElement('div');
    marker.id = 'global-drag-installed';
    marker.style.display = 'none';
    parentDoc.body.appendChild(marker);

    // Create subtle drag overlay
    const overlay = parentDoc.createElement('div');
    overlay.id = 'page-drag-overlay';
    overlay.innerHTML = `
        <div style="
            background: rgba(15, 23, 42, 0.92);
            border: 3px dashed #38BDF8;
            border-radius: 20px;
            padding: 36px 54px;
            text-align: center;
            box-shadow: 0 25px 60px rgba(0,0,0,0.6);
            backdrop-filter: blur(8px);
            pointer-events: none;
        ">
            <div style="font-size: 3rem; margin-bottom: 8px;">📥</div>
            <div style="color: #FFFFFF; font-size: 1.4rem; font-weight: 700; font-family: sans-serif;">
                Drop Metal Surface Image Anywhere
            </div>
            <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px; font-family: sans-serif;">
                Release to analyze surface for defects
            </div>
        </div>
    `;
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(10, 15, 29, 0.7);
        z-index: 9999999;
        display: none;
        justify-content: center;
        align-items: center;
        pointer-events: none;
    `;
    parentDoc.body.appendChild(overlay);

    let dragCount = 0;

    parentDoc.addEventListener('dragenter', function(e) {
        e.preventDefault();
        dragCount++;
        overlay.style.display = 'flex';
    });

    parentDoc.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dragCount--;
        if (dragCount <= 0) {
            dragCount = 0;
            overlay.style.display = 'none';
        }
    });

    parentDoc.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    parentDoc.addEventListener('drop', function(e) {
        e.preventDefault();
        dragCount = 0;
        overlay.style.display = 'none';

        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            // Dispatch to dropzone container
            const dropzone = parentDoc.querySelector('[data-testid="stFileUploaderDropzone"]');
            if (dropzone) {
                try {
                    const dropEvt = new DragEvent('drop', {
                        bubbles: true,
                        cancelable: true,
                        dataTransfer: e.dataTransfer
                    });
                    dropzone.dispatchEvent(dropEvt);
                } catch(err) {
                    console.warn(err);
                }
            }

            // Also set files on input element directly
            const fileInput = parentDoc.querySelector('section[data-testid="stFileUploader"] input[type="file"]') || 
                              parentDoc.querySelector('input[type="file"]');
            if (fileInput) {
                try {
                    fileInput.files = e.dataTransfer.files;
                    fileInput.dispatchEvent(new Event('input', { bubbles: true }));
                    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                } catch(err) {
                    console.warn(err);
                }
            }
        }
    });
})();
</script>
""", height=0, width=0)

if model is None:
    st.error("Model file not found. Please ensure 'models/defect_classifier.pth' exists.")
else:
    # -------------------------------------------------------------------------
    # SIMPLE INPUT SECTION
    # -------------------------------------------------------------------------
    col_input1, col_input2 = st.columns([1.5, 1])

    with col_input1:
        uploaded_file = st.file_uploader("Upload a metal image:", type=["jpg", "png", "jpeg", "bmp"])

    with col_input2:
        sample_choice = st.selectbox(
            "Or pick an example test image:",
            list(DEMO_SAMPLES.keys())
        )

    # Resolve selected image
    img_gray = None

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    elif DEMO_SAMPLES[sample_choice] is not None and os.path.exists(DEMO_SAMPLES[sample_choice]):
        img_gray = cv2.imread(DEMO_SAMPLES[sample_choice], cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------------------
    # RESULTS SECTION
    # -------------------------------------------------------------------------
    if img_gray is not None:
        img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
        img_3ch = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        with st.spinner("⚡ Running AI surface inspection & detecting anomalies..."):
            prob = predict(img_3ch)
            heatmap, dilated = create_heatmap(img_resized)

            if np.max(dilated) == 0:
                boxed = img_3ch.copy()
                regions = []
            else:
                boxed, regions = highlight_defects_with_regions(img_resized, heatmap)

        is_defect = prob > CONF_THRESHOLD

        # Big Clear Status Banner
        if is_defect:
            st.markdown(f"""
            <div class='result-card-defect'>
                ⚠️ DEFECT DETECTED — Risk Confidence: {prob * 100:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-normal'>
                ✅ NORMAL SURFACE — Safe Confidence: {(1 - prob) * 100:.1f}%
            </div>
            """, unsafe_allow_html=True)

        # 3 Quick Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Status", value="Defect Found" if is_defect else "Normal Surface")
        with m2:
            st.metric(label="Defect Probability", value=f"{prob * 100:.1f}%")
        with m3:
            st.metric(label="Defects Count", value=len(regions))

        st.write("")

        # 3 Visual Columns
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1. Original Image")
            st.image(img_resized, use_container_width=True)

        with c2:
            st.subheader("2. AI Heatmap")
            st.image(heatmap, use_container_width=True)

        with c3:
            st.subheader("3. Detected Defects")
            st.image(boxed, use_container_width=True)

        # Defect Details & Evidence (if any defects found)
        if regions:
            st.markdown("---")
            st.subheader("📋 Detected Defect Details")

            df = pd.DataFrame(regions)
            df.insert(0, "Defect #", range(1, len(regions) + 1))
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Defect #": st.column_config.NumberColumn(format="#%d"),
                    "Area": st.column_config.NumberColumn(format="%d px²"),
                }
            )

            st.subheader("🔎 Zoomed Defect Views")
            evidence_cols = st.columns(min(5, len(regions)))
            for idx, r in enumerate(regions):
                col_idx = idx % min(5, len(regions))
                x, y, w, h = r["X"], r["Y"], r["Width"], r["Height"]
                y1, y2 = max(0, y), min(img_resized.shape[0], y + h)
                x1, x2 = max(0, x), min(img_resized.shape[1], x + w)
                
                if y2 > y1 and x2 > x1:
                    crop = img_resized[y1:y2, x1:x2]
                    with evidence_cols[col_idx]:
                        st.image(crop, caption=f"Defect #{idx + 1}", use_container_width=True)

    else:
        st.info("👆 Please upload an image or choose an example from the dropdown above to inspect.")

# -----------------------------------------------------------------------------
# DISMISS PRELOADER WHEN ENTIRE PROJECT & UI HAVE FULLY LOADED
# -----------------------------------------------------------------------------
components.html("""
<script>
(function() {
    function tryDismiss() {
        if (typeof window.parent.dismissAIPagePreloader === 'function') {
            window.parent.dismissAIPagePreloader();
        } else {
            const parentDoc = window.parent.document;
            const preloader = parentDoc.getElementById('ai-page-preloader');
            if (preloader) {
                preloader.classList.add('fade-out');
                setTimeout(() => {
                    if (preloader && preloader.parentNode) {
                        preloader.parentNode.removeChild(preloader);
                    }
                }, 750);
            }
        }
    }
    setTimeout(tryDismiss, 350);
})();
</script>
""", height=0, width=0)
