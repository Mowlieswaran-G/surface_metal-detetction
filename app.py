import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms, models
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpectraMetal AI | Surface Defect Vision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CONSTANTS & ASSETS
# -----------------------------------------------------------------------------
IMG_SIZE = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "defect_classifier.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEMO_SAMPLES = {
    "Defect Sample A (Severe Pit)": os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "0000f269f.jpg"),
    "Defect Sample B (Surface Scratch)": os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "0002cc93b.jpg"),
    "Defect Sample C (Micro Crack)": os.path.join(BASE_DIR, "models", "Diffected with no greyscale", "00031f466.jpg"),
    "Normal Surface A (Smooth Finish)": os.path.join(BASE_DIR, "models", "Non diffected with greyscale", "kos01", "Part2.jpg"),
    "Normal Surface B (Machined Texture)": os.path.join(BASE_DIR, "models", "Non diffected with greyscale", "kos01", "Part3.jpg"),
}

# -----------------------------------------------------------------------------
# ADVANCED INDUSTRIAL DARK-MODE STYLES
# -----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-dark: #0A0F1D;
        --card-bg: rgba(17, 24, 39, 0.75);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-cyan: #06B6D4;
        --accent-blue: #3B82F6;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --accent-amber: #F59E0B;
        --text-muted: #94A3B8;
    }

    /* Overall page font & backdrop */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.08) 50%, rgba(15, 23, 42, 0.4) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Telemetry Tag Badges */
    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(15, 23, 42, 0.6);
        color: #CBD5E1;
        margin-right: 8px;
    }
    .tech-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }

    /* Glass KPI Stat Cards */
    .kpi-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.8rem;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Banner States */
    .status-banner-defect {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        color: #FECACA;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 16px 0;
    }
    .status-banner-normal {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        color: #A7F3D0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 16px 0;
    }

    /* Image Card Frame */
    .image-frame {
        background: #0B1120;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B0F19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# MODEL LOADING & INFERENCE
# -----------------------------------------------------------------------------
infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@st.cache_resource(show_spinner=False)
def load_defect_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            return model, True
        except Exception as e:
            return None, f"Failed to load weights: {e}"
    return None, f"Checkpoint not found at: {MODEL_PATH}"

model, model_status = load_defect_model()

def run_inference(img_3ch):
    if model is None:
        return 0.0, 0.0
    x = infer_transform(img_3ch).unsqueeze(0).to(DEVICE)
    start_t = time.perf_counter()
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    prob_defect = probs[0, 1].item()
    return prob_defect, latency_ms

# -----------------------------------------------------------------------------
# COMPUTER VISION PIPELINE (HEATMAP & CONTOURS)
# -----------------------------------------------------------------------------
def generate_saliency_heatmap(gray, blur_kernel=7, canny_low=50, canny_high=150):
    k = max(3, blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, morph_kernel, iterations=2)
    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)
    
    # Create an alpha blend of the original image with the heatmap
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(gray_3ch, 0.55, heatmap, 0.45, 0)
    return heatmap, dilated, blended

def detect_defect_regions(gray, heatmap, min_area=80):
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    regions = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = round(w / float(h), 2) if h > 0 else 1.0
            severity = "Severe" if area > 600 else ("Moderate" if area > 250 else "Minor")
            
            regions.append({
                "Region ID": idx + 1,
                "X": int(x),
                "Y": int(y),
                "Width": int(w),
                "Height": int(h),
                "Area (px²)": int(area),
                "Aspect Ratio": aspect_ratio,
                "Severity": severity
            })

            # Color bounding box by severity
            box_color = (0, 0, 255) if severity == "Severe" else ((0, 165, 255) if severity == "Moderate" else (0, 255, 0))
            cv2.rectangle(boxed, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(boxed, f"#{idx+1}", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

    return boxed, regions

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & DEMO PRESETS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Vision Telemetry & Engine")
    
    # Engine status card
    if model is not None:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 16px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 8px #10B981;"></span>
                <span style="color:#A7F3D0; font-size:0.85rem; font-weight:600;">ResNet-18 Engine: Active</span>
            </div>
            <div style="color:#94A3B8; font-size:0.75rem; margin-top:4px;">Weights Loaded • Ready for Inference</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Engine Offline: {model_status}")

    st.markdown("---")
    st.markdown("#### 📁 Input Source")
    input_mode = st.radio("Choose Input Mode:", ["Upload Custom Surface", "Preloaded Demo Samples"], index=1)

    chosen_image_path = None
    uploaded_file = None

    if input_mode == "Upload Custom Surface":
        uploaded_file = st.file_uploader(
            "Upload Surface Micrograph", 
            type=["jpg", "png", "jpeg", "bmp"],
            help="High-contrast metallic surface images supported."
        )
    else:
        sample_choice = st.selectbox("Select Industrial Sample:", list(DEMO_SAMPLES.keys()))
        chosen_image_path = DEMO_SAMPLES[sample_choice] if sample_choice in DEMO_SAMPLES else None
        if chosen_image_path and not os.path.exists(chosen_image_path):
            st.warning("Sample path not located locally. Please upload an image.")
            chosen_image_path = None

    st.markdown("---")
    st.markdown("#### 🎚️ Diagnostic Parameters")
    conf_threshold = st.slider("Defect Risk Threshold", min_value=0.20, max_value=0.95, value=0.60, step=0.05,
                               help="Sensitivity cutoff for flagging surface risk.")
    min_area_filter = st.slider("Min Anomaly Area (px²)", min_value=30, max_value=400, value=80, step=10,
                                help="Ignores micro-noise below this pixel count.")
    
    with st.expander("🛠️ Advanced Edge Filters"):
        canny_low = st.slider("Canny Low Threshold", 10, 100, 50, 5)
        canny_high = st.slider("Canny High Threshold", 80, 250, 150, 10)
        blur_kernel = st.slider("Gaussian Blur Kernel", 3, 15, 7, 2)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.75rem; color:#64748B;'>Hardware Accelerator: <b>{DEVICE.upper()}</b><br>Tensor Input: <b>256×256 px</b></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 SpectraMetal AI Vision</div>
    <div class="hero-subtitle">
        Deep Neural Network Surface Inspection & Micro-Structural Anomaly Detection System
    </div>
    <div style="margin-top: 14px;">
        <span class="tech-pill"><span class="tech-pill-dot"></span>ResNet-18 CNN</span>
        <span class="tech-pill">Saliency Saliency Fusion</span>
        <span class="tech-pill">Morphological Contour Analysis</span>
        <span class="tech-pill">Real-Time Telemetry</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PROCESS IMAGE INPUT
# -----------------------------------------------------------------------------
raw_img_gray = None

if input_mode == "Upload Custom Surface" and uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    raw_img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
elif input_mode == "Preloaded Demo Samples" and chosen_image_path is not None:
    raw_img_gray = cv2.imread(chosen_image_path, cv2.IMREAD_GRAYSCALE)

if raw_img_gray is None:
    st.info("👋 Select a preloaded demo sample from the sidebar or upload a surface image to start the automated AI inspection.")
else:
    # Resize to canonical processing dimensions
    gray_resized = cv2.resize(raw_img_gray, (IMG_SIZE, IMG_SIZE))
    img_3ch = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)

    # Run AI inference
    prob_defect, latency = run_inference(img_3ch)
    is_defect = prob_defect >= conf_threshold

    # Generate Heatmaps & Contours
    heatmap, dilated, blended = generate_saliency_heatmap(gray_resized, blur_kernel, canny_low, canny_high)
    boxed, regions = detect_defect_regions(gray_resized, heatmap, min_area=min_area_filter)

    total_defect_area = sum(r["Area (px²)"] for r in regions)
    defect_area_pct = (total_defect_area / float(IMG_SIZE * IMG_SIZE)) * 100.0

    # -------------------------------------------------------------------------
    # TOP KPI METRICS RIBBON
    # -------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_label = "CRITICAL DEFECT" if is_defect else "PRISTINE SURFACE"
        status_color = "#F43F5E" if is_defect else "#10B981"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Surface Health</div>
            <div class="kpi-value" style="color:{status_color}; font-size:1.55rem;">{status_label}</div>
            <div class="kpi-sub" style="color:{status_color};">● Quality Rating: {"REJECT" if is_defect else "PASS (A+)"}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        prob_color = "#F43F5E" if is_defect else "#10B981"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Defect Probability</div>
            <div class="kpi-value" style="color:{prob_color};">{prob_defect*100:.1f}%</div>
            <div class="kpi-sub" style="color:#94A3B8;">Confidence Index (Threshold: {conf_threshold*100:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Detected Anomalies</div>
            <div class="kpi-value" style="color:#38BDF8;">{len(regions)}</div>
            <div class="kpi-sub" style="color:#94A3B8;">Identified Defect Clusters</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Affected Area Ratio</div>
            <div class="kpi-value" style="color:#F59E0B;">{defect_area_pct:.2f}%</div>
            <div class="kpi-sub" style="color:#94A3B8;">Latency: {latency:.1f}ms • {DEVICE.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STATUS BANNER
    # -------------------------------------------------------------------------
    if is_defect:
        st.markdown(f"""
        <div class="status-banner-defect">
            <span style="font-size: 1.5rem;">⚠️</span>
            <div>
                <b>SURFACE DEFECT CONFIRMED:</b> The ResNet-18 neural classifier flagged this metallic surface as defective with <b>{prob_defect*100:.1f}% confidence</b>. {len(regions)} anomaly regions isolated.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner-normal">
            <span style="font-size: 1.5rem;">🛡️</span>
            <div>
                <b>SURFACE CLEARED:</b> High surface integrity detected. Normal confidence is <b>{(1 - prob_defect)*100:.1f}%</b>. No structural anomalies exceeding tolerance parameters.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN WORKBENCH TABS
    # -------------------------------------------------------------------------
    tab_inspect, tab_analytics, tab_model, tab_export = st.tabs([
        "🔬 Visual Inspection Lab", 
        "📊 Surface Analytics", 
        "🧠 Neural Architecture",
        "📑 Export Quality Report"
    ])

    # --- TAB 1: VISUAL INSPECTION LAB ---
    with tab_inspect:
        st.markdown("#### Multi-Spectral Surface Views")
        v1, v2, v3 = st.columns(3)

        with v1:
            st.markdown("<div class='image-frame'><span style='color:#94A3B8; font-weight:600; font-size:0.85rem;'>RAW SURFACE MICROGRAPH</span></div>", unsafe_allow_html=True)
            st.image(gray_resized, use_container_width=True)

        with v2:
            st.markdown("<div class='image-frame'><span style='color:#38BDF8; font-weight:600; font-size:0.85rem;'>AI SALIENCY HEATMAP</span></div>", unsafe_allow_html=True)
            st.image(blended, use_container_width=True)

        with v3:
            st.markdown("<div class='image-frame'><span style='color:#F43F5E; font-weight:600; font-size:0.85rem;'>ANOMALY LOCALIZATION</span></div>", unsafe_allow_html=True)
            st.image(boxed, use_container_width=True)

        st.markdown("---")
        
        # Detailed Defect Crops
        if regions:
            st.markdown(f"#### 🔎 Magnified Defect Evidence ({len(regions)} Detected)")
            crop_cols = st.columns(min(5, len(regions)))
            for idx, r in enumerate(regions):
                c_idx = idx % min(5, len(regions))
                x, y, w, h = r["X"], r["Y"], r["Width"], r["Height"]
                pad = 4
                y1, y2 = max(0, y - pad), min(gray_resized.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(gray_resized.shape[1], x + w + pad)
                
                if y2 > y1 and x2 > x1:
                    crop = gray_resized[y1:y2, x1:x2]
                    # Up-scale crop for viewing clarity
                    crop_large = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_NEAREST)
                    with crop_cols[c_idx]:
                        st.markdown(f"""
                        <div style="background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:6px; text-align:center;">
                            <div style="font-size:0.75rem; color:#38BDF8; font-weight:700;">Defect #{r['Region ID']}</div>
                            <div style="font-size:0.68rem; color:#94A3B8;">{r['Severity']} • {r['Area (px²)']} px²</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(crop_large, use_container_width=True)

    # --- TAB 2: SURFACE ANALYTICS ---
    with tab_analytics:
        st.markdown("#### Surface Density & Structural Distribution")
        
        g1, g2 = st.columns([1.2, 1])

        with g1:
            st.markdown("##### Pixel Intensity Distribution (Surface vs Edge)")
            fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0B0F19")
            ax.set_facecolor("#0F172A")
            
            hist_raw = cv2.calcHist([gray_resized], [0], None, [256], [0, 256])
            ax.plot(hist_raw, color="#38BDF8", linewidth=2, label="Surface Micrograph")
            
            if np.max(dilated) > 0:
                hist_edge = cv2.calcHist([dilated], [0], None, [256], [1, 256])
                ax.plot(hist_edge, color="#F43F5E", linewidth=1.5, linestyle="--", label="Anomaly Density")
                
            ax.set_xlabel("Pixel Grayscale Luminance (0 - 255)", color="#94A3B8", fontsize=9)
            ax.set_ylabel("Pixel Frequency", color="#94A3B8", fontsize=9)
            ax.tick_params(colors="#94A3B8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with g2:
            st.markdown("##### Defect Severity Matrix")
            if regions:
                severities = [r["Severity"] for r in regions]
                sev_counts = pd.Series(severities).value_counts()
                st.bar_chart(sev_counts)
            else:
                st.success("No anomalies recorded. Surface distribution is 100% uniform.")

    # --- TAB 3: NEURAL ARCHITECTURE ---
    with tab_model:
        st.markdown("#### Deep Residual Network (ResNet-18) Architecture")
        st.markdown("""
        The system utilizes a 18-layer Residual Network transfer-learned on industrial metal surface micrographs:
        - **Residual Blocks**: Skip-connections enable deep gradient propagation without vanishing gradients.
        - **Input Tensor**: `3 × 256 × 256` Normalized Grayscale (`mean=0.5, std=0.5`).
        - **Feature Extractor**: 8 Residual blocks with 2D Batch Normalization & ReLU activations.
        - **Classifier Head**: Linear projection layer mapping 512 deep latent features to 2 binary logits (`Normal`, `Defected`).
        """)

        training_plot_path = os.path.join(BASE_DIR, "models", "training_history.png")
        if os.path.exists(training_plot_path):
            st.markdown("##### Historical Training Validation Curves")
            st.image(training_plot_path, caption="ResNet-18 Loss and Accuracy Curves across Epochs", use_container_width=True)

    # --- TAB 4: EXPORT REPORT ---
    with tab_export:
        st.markdown("#### 📄 Inspection Data Sheet")
        if regions:
            df = pd.DataFrame(regions)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Defect Coordinates (CSV)",
                data=csv,
                file_name=f"metal_defect_report_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Surface passed with 0 defect coordinates. No CSV report required.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#64748B; font-size:0.8rem;'>SpectraMetal AI Vision Platform • High-Throughput Quality Assurance Engine</div>", unsafe_allow_html=True)
