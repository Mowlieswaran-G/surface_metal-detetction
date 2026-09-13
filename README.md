---
title: Metal Surface Defect Detection
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Metal Surface Defect Detection System 🔍

An automated AI computer vision application for detecting and localizing anomalies on metal surfaces using PyTorch (ResNet-18) and OpenCV.

## Features
- **AI Classification**: Accurately classifies metal surfaces as *Defective* or *Normal (Clean)*.
- **Visual Defect Heatmap**: Highlights high-density defect zones using edge energy and color-mapped contours.
- **Defect Region Localization**: Auto-detects bounding boxes and area calculations for localized defects.
- **Interactive UI**: Drag-and-drop support, built-in sample test images, and detailed CSV telemetry reports.

## Dataset Links
- [Defective Dataset](https://drive.google.com/file/d/1dNjG-DiPHDRBEtdOFLDBJdX0Ik_TMil9/view?usp=drive_link)
- [Non-Defective Dataset](https://drive.google.com/file/d/1eAQ4Pj-A4BM9FwZdEzauwrpkVYW4_66Z/view?usp=drive_link)
