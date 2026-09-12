
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms, models
import torch.nn.functional as F

st.set_page_config(
    page_title="Metal Defect Detection", 
    layout="wide"
)

IMG_SIZE = 256
MODEL_PATH = "models/defect_classifier.pth"
CONF_THRESHOLD = 0.6
device = "cpu"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        return None 
    model.eval()
    return model

model = load_model()

def predict(img_3ch):
    if model is None: return 0.0
    x = transform(img_3ch).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)
    return prob[0, 1].item()

def create_heatmap(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    heatmap = cv2.applyColorMap(dilated, cv2.COLORMAP_JET)
    return heatmap, dilated  # Return dilated for checking single-color

def highlight_defects_with_regions(gray, heatmap):
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({"X": x, "Y": y, "Width": w, "Height": h, "Area": area})
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output, regions

def local_css():
    st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px;}
    .sub-text {font-size: 1.1rem; color: #aaaaaa; margin-bottom: 20px;}
    .card {
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #1E88E5; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

st.title("Metal Surface Defect Detection")

if model is None:
    st.error("Model file not found. Please check 'models/defect_classifier.pth'.")
else:
    uploaded_file = st.file_uploader("Upload Surface Image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file:
        img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        prob = predict(img_3ch)
        heatmap, dilated = create_heatmap(img)

        # Check if heatmap is single-colored (all zeros)
        if np.max(dilated) == 0:
            st.warning("No detectable defect features in the image.")
            st.image(img, caption="Input Image", width=IMG_SIZE)
        else:
            boxed, regions = highlight_defects_with_regions(img, heatmap)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Analysis Status", "Complete")
            with m2:
                if prob > CONF_THRESHOLD:
                    st.metric("Defect Probability", f"{prob*100:.1f}%", "High Risk", delta_color="inverse")
                else:
                    st.metric("Defect Probability", f"{prob*100:.1f}%", "Safe", delta_color="normal")
            with m3:
                st.metric("Defects Found", len(regions))

            st.divider()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Original Input")
                st.image(img, width=IMG_SIZE)
            with c2:
                st.subheader("AI Heatmap")
                st.image(heatmap, width=IMG_SIZE)
            with c3:
                st.subheader("Highlighted Defects")
                st.image(boxed, width=IMG_SIZE)

            if prob > CONF_THRESHOLD:
                st.error(f"DEFECT DETECTED (Confidence: {prob*100:.1f}%)")
            else:
                st.success(f"NORMAL SURFACE (Confidence: {(1-prob)*100:.1f}%)")

            st.divider()

            if regions:
                st.subheader("Detailed Defect Report")
                df = pd.DataFrame(regions)
                df.insert(0, "Region ID", range(1, len(regions) + 1))
                st.dataframe(
                    df, 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Region ID": st.column_config.NumberColumn(format="#%d"),
                        "Area": st.column_config.NumberColumn(format="%d px²"),
                    }
                )

                st.subheader("Visual Evidence")
                cols = st.columns(5)
                for idx, region in enumerate(regions):
                    col_idx = idx % 5
                    x, y, w, h = region["X"], region["Y"], region["Width"], region["Height"]
                    y1, y2 = max(0, y), min(img.shape[0], y+h)
                    x1, x2 = max(0, x), min(img.shape[1], x+w)
                    if y2 > y1 and x2 > x1:
                        crop = img[y1:y2, x1:x2]
                        with cols[col_idx]:
                            st.image(crop, caption=f"Defect #{idx+1}", use_container_width=True)
            elif prob > CONF_THRESHOLD:
                st.warning("High defect probability detected.")
            else:
                st.info("No defect regions identified.")

    else:
        st.info("Please upload an image to start the inspection.")
