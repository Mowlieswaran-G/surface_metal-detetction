# 🔍 SurfaceAI — Metal Surface Defect Detection System

> **An AI-powered computer vision system for automated metal surface inspection**, designed to classify metal surfaces as **Normal** or **Defective** and provide visual defect analysis using a trained **ResNet-18 deep learning model** and **OpenCV-based image processing**.

SurfaceAI is an intelligent visual inspection application that helps automate the detection of defects in metal surfaces. Users can upload an image of a metal surface or select a built-in sample image, after which the system performs AI-based classification and generates visual analysis including heatmaps, defect regions, bounding boxes, and defect measurements.

The application is built with **Python, Streamlit, PyTorch, TorchVision, OpenCV, NumPy, and Pandas**, and can be deployed as a web application using **Streamlit Community Cloud**.

🔗 **Live App:** [surfaceai.streamlit.app](https://surfaceai.streamlit.app/)
📂 **Repository:** [github.com/Mowlieswaran-G/surface_metal-detetction](https://github.com/Mowlieswaran-G/surface_metal-detetction)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#overview)
- [What SurfaceAI Does](#what-surfaceai-does)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [AI Model](#ai-model)
- [Image Preprocessing](#image-preprocessing)
- [Defect Detection Pipeline](#defect-detection-pipeline)
- [Heatmap Generation](#heatmap-generation)
- [Defect Region Localization](#defect-region-localization)
- [Results and Analytics](#results-and-analytics)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Example Workflow](#example-workflow)
- [User Interface](#user-interface)
- [Core Components](#core-components)
- [Security Considerations](#security-considerations)
- [Potential Industrial Applications](#potential-industrial-applications)
- [Advantages](#advantages)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Testing Strategy](#testing-strategy)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Author](#author)
- [Project Summary](#project-summary)

---

<a id="overview"></a>
## 🧠 Overview

SurfaceAI is a computer vision application developed to assist in the inspection of metal surfaces.

Traditional visual inspection often requires a human operator to examine large numbers of manufactured components for scratches, pits, cracks, and other surface abnormalities. This can be time-consuming and may be affected by lighting conditions, fatigue, human error, and inspection workload.

SurfaceAI provides an AI-assisted alternative by combining:

- Deep learning-based image classification
- Image preprocessing
- Grayscale surface analysis
- OpenCV-based defect visualization
- Heatmap generation
- Defect region detection
- Bounding-box localization
- Interactive web-based inspection

The system accepts an image of a metal surface and produces an inspection result indicating whether the surface is **Normal** or **Defective**.

---

<a id="what-surfaceai-does"></a>
## 🔎 What SurfaceAI Does

```
                    Metal Surface Image
                             │
                             ▼
                   ┌───────────────────┐
                   │  Image Upload /   │
                   │   Example Image   │
                   └───────────────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │       Image       │
                   │   Preprocessing   │
                   └───────────────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │     ResNet-18     │
                   │   AI Classifier   │
                   └───────────────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │  Classification   │
                   └───────────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
           Normal Surface         Defective Surface
                  │                       │
                  │                       ▼
                  │               Heatmap Analysis
                  │                       │
                  │                       ▼
                  │              Region Detection
                  │                       │
                  │                       ▼
                  │                Bounding Boxes
                  │
                  └───────────┬───────────┘
                              ▼
                     Inspection Report
```

---

<a id="key-features"></a>
## ✨ Key Features

### 1. 🤖 AI-Based Surface Classification
SurfaceAI uses a trained ResNet-18 convolutional neural network to classify metal surface images into two categories:

- **Class 0** → Normal Surface
- **Class 1** → Defected Surface

The model outputs class probabilities which are used to calculate the defect probability displayed to the user.

### 2. 🖼️ Image Upload
Users can upload their own metal surface images directly through the web interface. Supported formats:
- JPG
- JPEG
- PNG
- BMP

### 3. 🧪 Built-In Sample Images
The application includes sample images so users can test the system without uploading their own images:
- Defect — Pit / Scratch
- Defect — Crack
- Normal — Clean Metal Surface

### 4. 🔥 AI-Assisted Heatmap Visualization
After image analysis, SurfaceAI generates a heatmap highlighting areas with stronger surface irregularities, helping users understand where the system is detecting surface activity.

### 5. 📦 Defect Region Localization
SurfaceAI processes the generated heatmap to identify potential defect regions and calculates:
- Bounding box coordinates
- Width
- Height
- Area
- Region location

### 6. 🔎 Zoomed Defect Views
Detected regions can be displayed as individual cropped images for closer inspection.

### 7. 📊 Inspection Metrics
Example output:
```
Status: Defect Found
Defect Probability: 94.6%
Defects Count: 3
```

### 8. 📋 Defect Telemetry
Structured information about detected regions for inspection analysis and further processing.

---

<a id="architecture-overview"></a>
## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      SurfaceAI Web UI                       │
│                          Streamlit                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Image Processing Layer                    │
│           OpenCV + NumPy + Image Transformations            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI Model Layer                        │
│                          ResNet-18                          │
│                    PyTorch / TorchVision                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Analysis Layer                        │
│         Classification + Heatmap + Region Detection         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         Results UI                          │
│         Original Image | Heatmap | Detected Regions         │
│         Confidence | Defect Count | Region Details          │
└─────────────────────────────────────────────────────────────┘
```

---

<a id="ai-model"></a>
## 🧬 AI Model

The core classification model is based on **ResNet-18**, a convolutional neural network architecture that uses residual connections to enable effective training of deep neural networks.

```
ResNet-18 Backbone
        │
        ▼
Feature Extraction
        │
        ▼
Fully Connected Layer
        │
        ▼
2 Output Classes
   ┌──────────────┐
   │ Normal       │
   │ Defected     │
   └──────────────┘
```

The trained model is stored at `models/defect_classifier.pth` and loaded during application startup.

### ⚙️ Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | ResNet-18 |
| Framework | PyTorch |
| Classes | 2 |
| Class 0 | Normal |
| Class 1 | Defected |
| Input Size | 256 × 256 |
| Image Representation | 3-channel grayscale |
| Loss Function | Cross Entropy Loss |
| Optimizer | Adam |
| Weight Decay | 1e-5 |

---

<a id="image-preprocessing"></a>
## 🖼️ Image Preprocessing

```
Input Image
     │
     ▼
Resize to 256 × 256
     │
     ▼
Grayscale Processing
     │
     ▼
Convert to 3 Channels
     │
     ▼
Tensor Conversion
     │
     ▼
Normalization
     │
     ▼
ResNet-18
```

Normalization used during training and inference:
```
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

---

<a id="defect-detection-pipeline"></a>
## 🔬 Defect Detection Pipeline

**Stage 1 — AI Classification**
The ResNet-18 model determines whether the uploaded image is Normal or Defected, and outputs class probabilities.

**Stage 2 — Visual Defect Analysis**
For defect visualization, the application performs additional image processing using OpenCV. The image is analyzed for surface irregularities and converted into a heatmap, from which potential defect regions are extracted.

---

<a id="heatmap-generation"></a>
## 🔥 Heatmap Generation

```
Grayscale Image
       │
       ▼
Image Processing
       │
       ▼
Intensity / Edge Analysis
       │
       ▼
Morphological Processing
       │
       ▼
Color Mapping
       │
       ▼
Defect Heatmap
```

---

<a id="defect-region-localization"></a>
## 📦 Defect Region Localization

After generating the heatmap, SurfaceAI analyzes the processed regions and identifies contours. For each detected contour, the system calculates a bounding rectangle `(x, y, width, height)`, which is then visualized on the original image.

```
┌──────────────────────────────┐
│                              │
│        ┌────────────┐        │
│        │  DEFECT    │        │
│        │   REGION   │        │
│        └────────────┘        │
│                              │
└──────────────────────────────┘
```

---

<a id="results-and-analytics"></a>
## 📊 Results and Analytics

**Classification Result:** `Normal Surface` or `Defect Found`

**Defect Probability:** e.g. `91.4%`

**Defect Count:** e.g. `Defects Count: 4`

**Visual Analysis includes:**
- Original image
- AI heatmap
- Defect-region visualization
- Detailed defect information
- Zoomed defect crops

---

<a id="project-structure"></a>
## 📂 Project Structure

```
surface_metal-detetction/
│
├── app.py                     → Main Streamlit application
├── train_model.py             → Model training pipeline
├── requirements.txt           → Python dependencies
├── packages.txt               → Linux system packages required by deployment
├── README.md                  → Project documentation
├── models/
│   └── defect_classifier.pth  → Trained ResNet-18 model weights
├── samples/
│   ├── defect_crack.jpg
│   ├── defect_pit.jpg
│   └── normal_surface.jpg
├── train.log                  → Training logs
├── train_output.txt           → Training output
├── .gitignore
└── .vscode/
    └── settings.json
```

---

<a id="technology-stack"></a>
## 🛠️ Technology Stack

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/TorchVision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="TorchVision"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" alt="Matplotlib"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
</p>

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Deep Learning | PyTorch |
| CNN Architecture | ResNet-18 |
| Computer Vision | OpenCV |
| Neural Network Utilities | TorchVision |
| Numerical Processing | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| ML Utilities | Scikit-learn |
| Deployment | Streamlit Community Cloud |
| Model Format | PyTorch `.pth` |

### 📦 Python Dependencies (`requirements.txt`)
```
streamlit
torch
torchvision
opencv-python-headless
pandas
numpy
matplotlib
scikit-learn
```

---

<a id="dataset"></a>
## 🗃️ Dataset

The project uses separate datasets for **Defective Metal Surfaces** and **Non-Defective Metal Surfaces**, treated as a binary classification problem:

```
Normal     → 0
Defected   → 1
```

The dataset is processed into training and validation sets before model training.

**Dataset Source:**

| Class | Link |
|---|---|
| Defected | [Google Drive Folder](https://drive.google.com/drive/folders/1E6hlvQ_eErhnQLxvVMO9hB26hkwle_lG?usp=sharing) |
| Non-Defected | [Google Drive Folder](https://drive.google.com/drive/folders/1CDH2zvebl7PAKUGJLC4MQOxc85qekV72?usp=sharing) |

> **Note:** These are private Google Drive folders. Anyone cloning this repository will need access granted separately, or the images should be copied into a local `dataset/` directory (e.g. `dataset/defected/` and `dataset/non_defected/`) before running `train_model.py`.

---

<a id="model-training"></a>
## 🏋️ Model Training

Implemented in `train_model.py`, using ResNet-18 with a modified final layer for binary classification.

```
Dataset
   │
   ▼
Image Loading
   │
   ▼
Resize 256 × 256
   │
   ▼
Grayscale → 3 Channels
   │
   ▼
Normalization
   │
   ▼
Weighted Sampling
   │
   ▼
ResNet-18
   │
   ▼
Cross Entropy Loss
   │
   ▼
Adam Optimizer
   │
   ▼
Validation
   │
   ▼
Best Model
   │
   ▼
defect_classifier.pth
```

### ⚖️ Class Balancing
The training pipeline calculates class weights from the training labels. A weighted sampling strategy improves batch balance between the two classes — particularly useful when the number of normal and defective examples is not perfectly balanced.

### 🎯 Loss Function
`CrossEntropyLoss()` for binary classification with two output classes.

### ⚡ Optimizer
**Adam**, with:
- Learning Rate: `0.0002`
- Weight Decay: `1e-5`

### 🖥️ Training Hardware
Training was performed on **CPU**. The code automatically uses GPU acceleration if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

A learning-rate scheduler is also used to reduce the learning rate when validation performance stops improving.

### 💾 Model Output
The best-performing model is saved as `models/defect_classifier.pth`, which is loaded by the Streamlit application during inference.

### 📈 Model Performance

| Metric | Training Set (Final Epoch) | Validation Set (3,807 images) |
|---|---|---|
| Accuracy | 100.00% | 100.00% |
| Loss | 0.0001 | 0.0000 |
| Precision | ~100% (1.00) | 100.00% (1.00) |
| Recall | ~100% (1.00) | 100.00% (1.00) |
| F1-Score | ~100% (1.00) | 100.00% (1.00) |

> **Note:** These near-perfect scores likely reflect strong separability within the current dataset (e.g. consistent imaging conditions, limited visual diversity) rather than guaranteed real-world performance. As noted in [Current Limitations](#-current-limitations), the model should be validated on independent, more diverse production images before being trusted for industrial quality decisions — 100% validation accuracy is a signal to check for dataset overlap or low variability, not a guarantee of generalization.

---

<a id="installation"></a>
## 💻 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mowlieswaran-G/surface_metal-detetction.git
cd surface_metal-detetction
```

### 2. Create a Virtual Environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

<a id="running-locally"></a>
## ▶️ Running Locally

```bash
streamlit run app.py
```

The application will normally become available at:
```
http://localhost:8501
```

---

<a id="deployment"></a>
## 🌐 Deployment

SurfaceAI can be deployed as a Streamlit web application. The repository includes Streamlit configuration metadata and deployment dependencies. For Streamlit Community Cloud, the main entry point is `app.py`.

```
GitHub Repository
        │
        ▼
Streamlit Community Cloud
        │
        ▼
Install requirements.txt
        │
        ▼
Load app.py
        │
        ▼
Load defect_classifier.pth
        │
        ▼
SurfaceAI Web Application
```

---

<a id="example-workflow"></a>
## 🧪 Example Workflow

1. **Open SurfaceAI** — launch the application.
2. **Upload an Image** — supported formats: JPG, JPEG, PNG, BMP.
3. **AI Classification** — the ResNet-18 model analyzes the image and produces `Normal Surface` or `Defect Found`.
4. **Inspect the Heatmap** — SurfaceAI generates a heatmap showing areas of interest.
5. **Examine Defect Regions** — candidate regions are identified with bounding boxes.
6. **Review Metrics** — Status, Defect Probability, Defect Count, and region-level information.

**Workflow:** `Upload → Analyze → Visualize → Inspect`

---

<a id="user-interface"></a>
## 🖥️ User Interface

- Dark-themed interface
- AI loading animation
- Image upload
- Example image selection
- Classification result
- Defect probability
- Heatmap visualization
- Defect region visualization
- Defect count
- Detailed defect information
- Zoomed defect views

---

<a id="core-components"></a>
## 🧩 Core Components

**`app.py`** — Main application: Streamlit UI, image upload, example image selection, model loading, preprocessing, AI prediction, heatmap generation, defect localization, result visualization.

**`train_model.py`** — Dataset loading, preprocessing, training, validation, class balancing, optimization, model checkpointing, training history logging.

**`models/defect_classifier.pth`** — The trained PyTorch model used for inference.

**`samples/`** — Example images for testing: `defect_crack.jpg`, `defect_pit.jpg`, `normal_surface.jpg`.

---

<a id="security-considerations"></a>
## 🔐 Security Considerations

SurfaceAI is primarily an image-analysis prototype. When deploying publicly:

- Do not upload private or confidential images.
- Do not commit API keys or passwords.
- Avoid storing sensitive inspection images unnecessarily.
- Review the hosting provider's data handling policies before using the system with confidential industrial data.

---

<a id="potential-industrial-applications"></a>
## 🏭 Potential Industrial Applications

- **Manufacturing** — Inspecting metal components during production.
- **Automotive** — Detecting visible surface abnormalities in manufactured metal parts.
- **Industrial Equipment** — Inspecting metal panels and components.
- **Quality Control** — Supporting quality-control teams during visual inspection.
- **Research & Prototyping** — Demonstrating AI-assisted industrial computer vision systems.
- **Educational Applications** — Teaching deep learning and computer vision using a practical industrial use case.

---

<a id="advantages"></a>
## 🚀 Advantages

- **Automated Inspection** — Reduces dependence on completely manual visual inspection.
- **Fast Analysis** — Rapid inference on an uploaded image.
- **Visual Explainability** — Heatmaps and localized regions add visual context to classification.
- **Interactive Interface** — Browser-based workflow, no Python interaction needed.
- **Easy Demonstration** — Built-in sample images.
- **Deployable Architecture** — Hostable via Streamlit Community Cloud.

---

<a id="current-limitations"></a>
## ⚠️ Current Limitations

1. **Binary Classification** — Distinguishes only Normal vs. Defected; no dedicated class per defect type.
2. **Heatmap Is Not a Trained Object Detector** — Regions are image-processing candidates, not outputs of a dedicated detection model.
3. **Dataset Dependence** — Performance depends on dataset quality, image diversity, lighting, surface types, camera setup, and defect characteristics.
4. **Real-World Validation** — Needs evaluation on representative production images and validation against established QC procedures.
5. **Camera and Lighting Conditions** — Illumination, camera angle, resolution, reflections, and surface finish can affect performance.

---

<a id="future-improvements"></a>
## 🔮 Future Improvements

1. **Multi-Class Defect Classification** — Normal, Crack, Pit, Scratch, Corrosion, Dent, Surface Contamination.
2. **Dedicated Object Detection** — YOLO, Faster R-CNN, Mask R-CNN, YOLO Segmentation.
3. **Pixel-Level Segmentation** — Precise defect shape and boundary identification.
4. **Improved Explainability** — Grad-CAM, Grad-CAM++, Integrated Gradients.
5. **Real-Time Camera Inspection**:
   ```
   Industrial Camera → Frame Capture → SurfaceAI → AI Classification
   → Defect Localization → Quality Control Decision
   ```
6. **Production Quality Dashboard** — Inspection history, defect statistics, batch analysis, frequency, confidence trends, exportable reports.
7. **Model Optimization** — ONNX, TorchScript, Quantization, GPU inference, Batch inference.

### 📈 Future Architecture

```
       Industrial Camera
               │
               ▼
  ┌─────────────────────────┐
  │    Image Acquisition    │
  └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │ Preprocessing (OpenCV)  │
  └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │    AI Vision Engine     │
  │     ResNet / YOLO /     │
  │   Segmentation Model    │
  └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │     Defect Analysis     │
  └─────────────────────────┘
               │
          ┌─────┴─────┐
          ▼           ▼
         PASS       FAIL
          │           │
          ▼           ▼
     Production    Quality
      Continue   Inspection
```

---

<a id="testing-strategy"></a>
## 🧪 Testing Strategy

**Classification:** Accuracy, Precision, Recall, F1-score, Confusion matrix

**Defect Localization:** Intersection over Union (IoU), Detection precision, Detection recall

**Production Performance:** Inference time, Throughput, False positive rate, False negative rate

> For industrial inspection, reducing false negatives is particularly important because missed defects may reach downstream production stages.

---

<a id="project-status"></a>
## 📌 Project Status

| | |
|---|---|
| Project Type | AI Computer Vision Prototype |
| Current Task | Binary Metal Surface Classification |
| Model | ResNet-18 |
| Classes | Normal / Defected |
| Interface | Streamlit |
| Image Processing | OpenCV |
| Deployment | Streamlit Community Cloud |
| Status | Working Prototype |

---

<a id="contributing"></a>
## 🤝 Contributing

Contributions and improvements are welcome. Possible contribution areas include:

- Model improvements
- Dataset expansion
- Defect segmentation
- UI improvements
- Performance optimization
- Deployment improvements
- Industrial camera integration
- Testing and evaluation

**To contribute:**
```bash
git clone https://github.com/Mowlieswaran-G/surface_metal-detetction.git
cd surface_metal-detetction
git checkout -b feature/your-feature
```
Make your changes, commit them, and open a pull request.

---

<a id="license"></a>
## 📜 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Mowlieswaran G

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

A copy of this license is also included in the repository as a standalone [`LICENSE`](./LICENSE) file.

---

<a id="disclaimer"></a>
## ⚠️ Disclaimer

SurfaceAI is an AI-based research and demonstration prototype. The predictions generated by the system should not be treated as a definitive industrial quality certification without proper validation against domain-specific inspection standards.

Before using the system in a production manufacturing environment, the model should be independently validated using representative production data, calibrated inspection equipment, and appropriate quality-control procedures.

---

<a id="author"></a>
## 👨‍💻 Author

**Mowlieswaran G**

- GitHub: https://github.com/Mowlieswaran-G
- Project: https://github.com/Mowlieswaran-G/surface_metal-detetction

---

<a id="project-summary"></a>
## ⭐ Project Summary

SurfaceAI demonstrates how deep learning and classical computer vision can be combined to create an intelligent metal surface inspection system, bringing together **PyTorch (ResNet-18 classifier)**, **OpenCV (heatmaps & localization)**, and **Streamlit (interactive UI)** into one workflow — turning metal surface images into actionable visual inspection insights using AI.
