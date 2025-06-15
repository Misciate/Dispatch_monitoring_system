import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from pathlib import Path

# --- Load YOLOv5 model ---
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='Models/yolov5/runs/train/dispatch_yolo/weights/best.pt')  # cập nhật path nếu khác
yolo_model.conf = 0.25  # confidence threshold
yolo_model.iou = 0.45   # iou threshold
yolo_model.classes = None  # detect cả dish và tray

# --- Load MobileNetV2 classification model ---
mobilenet_model = models.mobilenet_v2(weights=None)
mobilenet_model.classifier[1] = torch.nn.Linear(mobilenet_model.last_channel, 6)
mobilenet_model.load_state_dict(torch.load('Models/mobilenetv2_classification.pth', map_location='cpu'))  # đổi thành 'cuda' nếu dùng GPU
mobilenet_model.eval()

# --- Class mapping ---
class_mapping = {
    0: 'dish/empty',
    1: 'dish/kakigori',
    2: 'dish/not_empty',
    3: 'tray/empty',
    4: 'tray/kakigori',
    5: 'tray/not_empty'
}

# --- Preprocessing for classification ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Inference function ---
def detect_and_classify(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        boxes = results.xyxy[0]  # tensor: (x1, y1, x2, y2, conf, cls)

        for *xyxy, conf, cls in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            label_det = int(cls.item())

            # Crop region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Convert to PIL and classify
            try:
                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(img_pil).unsqueeze(0)
                output = mobilenet_model(input_tensor)
                class_idx = torch.argmax(output, 1).item()
                label = class_mapping[class_idx]
            except Exception as e:
                label = "Error"

            # Draw results
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1
        print(f"[Frame {frame_idx}] Processed")

    cap.release()
    out.release()
    print(f"✅ Output saved to: {output_path}")


# --- Main ---
if __name__ == '__main__':
    video_path = 'Dataset/example_1.mp4'  # ← Cập nhật đúng tên video của bạn
    output_path = 'output_result.mp4'
    detect_and_classify(video_path, output_path)
