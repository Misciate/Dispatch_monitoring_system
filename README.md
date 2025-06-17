# Dispatch Monitoring System

## Overview
The **Dispatch Monitoring System** is an intelligent monitoring solution designed for a commercial kitchen's dispatch area. It leverages computer vision techniques, utilizing YOLOv5 for object detection and MobileNetV2 for classification, to track items (dishes and trays) within a predefined Region of Interest (ROI). This project, developed as part of an interview test assignment, includes functionality to improve model performance based on user feedback through a Streamlit-based interface.

## Features
- **Object Detection**: Uses YOLOv5 to detect dishes and trays in the dispatch area.
- **Classification**: Employs MobileNetV2 to classify detected objects into categories: `dish/empty`, `dish/kakigori`, `dish/not_empty`, `tray/empty`, `tray/kakigori`, and `tray/not_empty`.
- **Video Processing**: Processes input video and generates annotated output video with bounding boxes and labels.
- **User Feedback Integration**: Allows users to provide feedback on classification results via a web interface to enhance model accuracy.
- **Model Retraining**: Supports retraining the classification model with user feedback data.

## Requirements
- **Hardware**:
  - CPU with virtualization support (Intel VT-x or AMD-V), enabled in BIOS.
  - Minimum 8GB RAM (16GB recommended for smooth Docker performance).
- **Software**:
  - [Docker](https://www.docker.com/products/docker-desktop/)
  - [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)
  - Git (to clone repository)

## Installation

### Prerequisites
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and ensure Docker Compose is available.
   - On Windows: Enable **Virtual Machine Platform** and **Hyper-V** (see below).
   - Check with: `docker --version` and `docker compose --version`.
2. Clone the repository:
   ```bash
   git clone https://github.com/Misciate/Dispatch_monitoring_system.git
   cd Dispatch-Monitoring-System