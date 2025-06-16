# Dispatch Monitoring System

## Overview
This project is an intelligent monitoring system designed for a commercial kitchen's dispatch area. It leverages computer vision techniques to track items (dishes and trays) within the dispatch area using a combination of YOLOv5 for object detection and MobileNetV2 for classification. The system is developed as part of an interview test assignment and includes functionality to improve model performance based on user feedback.

## Features
- **Object Detection**: Uses YOLOv5 to detect dishes and trays in a predefined Region of Interest (ROI) within the dispatch area.
- **Classification**: Employs MobileNetV2 to classify detected objects into categories such as `dish/empty`, `dish/kakigori`, `dish/not_empty`, `tray/empty`, `tray/kakigori`, and `tray/not_empty`.
- **Video Processing**: Processes video input and outputs annotated video with bounding boxes and labels.
- **ROI Optimization**: Limits detection to a specific area (e.g., the wooden counter near the kitchen) to match the training dataset.

## Requirements
- **Python 3.8+**
- **Dependencies**:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `ultralytics` (for YOLOv5)
- **Hardware**: GPU recommended for faster inference (optional).
- **Docker**: For deployment using Docker Compose.

## Installation

### Prerequisites
1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
2. Clone the repository:
   ```bash
   git clone <your-github-repo-link>
   cd <repository-name>
