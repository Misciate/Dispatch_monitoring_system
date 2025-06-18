# Dispatch Monitoring System

## Overview

The **Dispatch Monitoring System** is an intelligent computer vision-based monitoring solution designed for a commercial kitchen’s dispatch area. It combines **YOLOv5** for object detection and **MobileNetV2** for object classification. The system tracks items such as dishes and trays within a Region of Interest (ROI), enabling real-time annotation and feedback integration. Developed as part of a technical interview project, the system features a user-friendly interface for model refinement and retraining based on human feedback.

---

## Features

- 🔍 **Object Detection**: Detects dishes and trays using YOLOv5.
- 🧐 **Classification**: Classifies detected objects into:
  - `dish/empty`, `dish/kakigori`, `dish/not_empty`
  - `tray/empty`, `tray/kakigori`, `tray/not_empty`
- 🎮 **Video Processing**: Processes videos frame-by-frame and outputs annotated results.
- 💬 **Feedback Integration**: Collects user feedback through a Streamlit interface for error correction and retraining.
- 🔀 **Model Retraining**: Supports retraining the classification model using confirmed user feedback to improve performance over time.

---

## System Architecture & Workflow

1. **Input**: A raw video is fed into the system.
2. **YOLOv5** performs object detection, drawing bounding boxes around trays and dishes within the defined ROI.
3. **MobileNetV2** classifies each detected object.
4. The system overlays labels on the output video.
5. **User Feedback Interface** (Streamlit) allows corrections to classification results.
6. Confirmed feedback is stored in a structured folder hierarchy.
7. Feedback data is later used for **retraining** the classification model via the provided training script.

---

## Installation Guide

### 📁 Download Dataset

You can access the training dataset used in this project here:\
[Dataset Link (Google Drive)](https://drive.google.com/drive/folders/1chvJfXgbFI3GSSa-8bHJh7kbDbx4hPnp?usp=sharing)

I also finetune dataset by adding classification_flat folder to easier for training model.
When you done download, just pull the git repo then put the dataset folder into the system folder.

### 🛠️ Requirements

#### Hardware

- CPU with virtualization support (Intel VT-x / AMD-V enabled in BIOS)
- Minimum 8GB RAM (16GB recommended for smooth Docker usage)

#### Software

- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)
- Git (to clone the repository)

### 🚀 Installation Steps

1. **Install Docker & Docker Compose**

   - Make sure virtualization is enabled in BIOS.
   - Verify installation:
     ```bash
     docker --version
     docker compose --version
     ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/Misciate/Dispatch_monitoring_system.git
   cd Dispatch-Monitoring-System
   add the dataset folder into the Dispatch-Monitoring-System folder.
   ```

3. **Build and launch the app**

   ```bash
   docker compose up --build
   ```

   This will start the main services including the feedback interface.

4. **Access the Feedback Interface** Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
Dispatch-Monitoring-System/
├── Models/                      # Trained models (best.pt, mobilenetv2_classification.pth)
├── feedback/
│   ├── confirmed/              # Subdirectories by class for confirmed feedback
│   │   ├── dish/empty/
│   │   ├── dish/kakigori/
│   │   ├── dish/not_empty/
│   │   ├── tray/empty/
│   │   ├── tray/kakigori/
│   │   └── tray/not_empty/
├── detect_classificate_video.py  # Main detection and classification script
├── feedback_interface.py         # Streamlit app for feedback collection
├── retrain_model.py              # Retrain classifier using feedback data
├── train_classification.py       # Script to train classification model from scratch
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose definition
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore
```

---

## 🧠 Model Training Mechanism

### 🔍 Object Detection (YOLOv5)

- The object detection model is trained using the provided dataset, keeping the original `detection/` folder structure intact.
- YOLOv5 is configured to detect **two object classes**:
  - `0`: dish
  - `1`: tray
- The detection model outputs bounding boxes and class labels for each frame.

### 🚪 Object Classification (MobileNetV2)

- For classification, a separate folder structure named `classification_flat/` is created.
- It contains **six subfolders** corresponding to all object-category combinations:
  - `dish/empty`
  - `dish/kakigori`
  - `dish/not_empty`
  - `tray/empty`
  - `tray/kakigori`
  - `tray/not_empty`
- Cropped object images from detection outputs are used to populate this directory.
- A **MobileNetV2** model is trained on this folder using standard supervised learning.

### 🔗 Integration

- During inference:
  1. The YOLOv5 model detects the object and assigns a label (`0` for dish, `1` for tray).
  2. The corresponding cropped object is passed to the classification model.
  3. Depending on the detected class, the system maps the classification output to one of the 6 categories.

## 🎥 Detection & Classification in Restaurant Videos

Due to the nature of the original detection dataset—focused solely on tabletop regions with wooden textures—false positives may occur in irrelevant areas of the video. To mitigate this, the system applies the following:

### 📌 Region of Interest (ROI)

- A manually defined **Region of Interest (ROI)** is used to focus detection only on the dispatch table area.
- This filters out background objects and unrelated regions in the video frame.

### 📉 Confidence Threshold

- Classification results are filtered by **probability threshold**.
- Only predictions with a **confidence score ≥ 70%** are visualized and annotated in the final output video.
- This helps reduce visual clutter and prevents low-confidence predictions from misleading viewers.

## 🔄 Feedback Mechanism

The feedback workflow is designed to allow users to easily review and correct model predictions within the video stream:

1. Users watch the detection video via the **Streamlit interface**.
2. A **slider** allows navigation to any specific frame.
3. When users identify a potentially incorrect label, they can click **“Extract and Feedback”**.
   - This extracts the current frame and saves the cropped object to the "feedback_pending.txt" along with its predicted label.
4. The user then selects the correct label from a dropdown menu.
5. After submitting the correction, the feedback is:
   - Moved to the `feedback/confirmed/[correct_label]/` directory.
   - Logged in the `feedback.csv` file for auditing and retraining purposes.

This pipeline ensures **human-in-the-loop learning**, allowing the system to evolve based on real-world usage.

---

