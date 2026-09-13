import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
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
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
IMG_SIZE = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "defect_classifier.pth")
CONF_THRESHOLD = 0.60
DEVICE = "cpu"

DEMO_SAMPLES = {
    "None (Upload my own)": None,
    "Example 1: Surface Defect (Pit/Scratch)": os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "0000f269f.jpg"),
    "Example 2: Surface Defect (Crack)": os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "0002cc93b.jpg"),
    "Example 3: Normal Surface (Clean Metal)": os.path.join(BASE_DIR, "models", "Non diffected with greyscale", "kos01", "Part2.jpg"),
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
    /* Prominent Drag and Drop Zone */
    [data-testid="stFileUploader"] {
        width: 100%;
        margin-bottom: 12px;
    }
    [data-testid="stFileUploader"] section {
        padding: 36px 20px !important;
        border: 2px dashed #38BDF8 !important;
        border-radius: 16px !important;
        background: rgba(56, 189, 248, 0.04) !important;
        transition: all 0.25s ease-in-out !important;
        text-align: center;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #06B6D4 !important;
        background: rgba(56, 189, 248, 0.1) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2) !important;
    }
    [data-testid="stFileUploader"] section span {
        color: #F8FAFC !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] section small {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
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

if model is None:
    st.error("Model file not found. Please ensure 'models/defect_classifier.pth' exists.")
else:
    # -------------------------------------------------------------------------
    # SIMPLE INPUT SECTION (DRAG & DROP ZONE)
    # -------------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "📁 Drag and drop your metal surface image here, or click to browse",
        type=["jpg", "png", "jpeg", "bmp"],
        help="Upload any high-resolution surface image file"
    )

    with st.expander("💡 Or test with preloaded sample images from the dataset"):
        sample_choice = st.selectbox(
            "Choose a sample to inspect:",
            list(DEMO_SAMPLES.keys()),
            index=0
        )

    # Resolve selected image
    img_gray = None

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    elif sample_choice != "None (Upload my own)" and DEMO_SAMPLES.get(sample_choice) and os.path.exists(DEMO_SAMPLES[sample_choice]):
        img_gray = cv2.imread(DEMO_SAMPLES[sample_choice], cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------------------
    # RESULTS SECTION
    # -------------------------------------------------------------------------
    if img_gray is not None:
        img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
        img_3ch = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

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
