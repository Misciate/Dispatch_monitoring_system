import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Load YOLOv5 model ---
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='Models/best.pt')
yolo_model.conf = 0.25
yolo_model.iou = 0.45
yolo_model.classes = None

# --- Load MobileNetV2 classification model ---
mobilenet_model = models.mobilenet_v2(weights=None)
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, 6)
mobilenet_model.load_state_dict(torch.load('Models/mobilenetv2_classification.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
mobilenet_model.eval()

# --- Unified class mapping ---
class_names = {
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Detection + Classification with ROI, validation, and confidence threshold ---
def detect_and_classify(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- Define ROI: Adjust here if needed ---
    roi_x, roi_y = 1300, 98
    roi_width, roi_height = 2000, 400

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Khởi tạo thư mục feedback nếu chưa tồn tại
    feedback_dir = Path("feedback")
    feedback_dir.mkdir(exist_ok=True)
    feedback_pending_file = "feedback_pending.txt"
    if os.path.exists(feedback_pending_file):
        os.remove(feedback_pending_file)  # Xóa file cũ để tránh trùng lặp

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop ROI
        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        if roi.size == 0:
            print(f"[⚠️] ROI empty at frame {frame_idx}")
            out.write(frame)
            frame_idx += 1
            continue

        # --- YOLO Detection on ROI ---
        results = yolo_model(roi)
        boxes = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

        for *xyxy, conf, cls in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            x1_full, y1_full = roi_x + x1, roi_y + y1
            x2_full, y2_full = roi_x + x2, roi_y + y2

            crop = roi[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                print(f"[⚠️] Invalid crop at frame {frame_idx}, size: {crop.shape}")
                continue

            try:
                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(img_pil).unsqueeze(0)

                with torch.no_grad():
                    output = mobilenet_model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    max_prob, class_idx = torch.max(probabilities, dim=1)
                    max_prob = max_prob.item()
                    class_idx = class_idx.item()

                # Validate class index based on YOLO label
                yolo_label = int(cls.item())
                if (yolo_label == 0 and class_idx not in [0, 1, 2]) or (yolo_label == 1 and class_idx not in [3, 4, 5]):
                    print(f"[⚠️] Mismatch: YOLO {yolo_label} with Class {class_idx} at frame {frame_idx}, skipping...")
                    continue

                # Check confidence threshold
                if max_prob < 0.70:
                    print(f"[⚠️] Low confidence {max_prob:.2f} at frame {frame_idx}, skipping...")
                    continue

                if class_idx not in class_names:
                    print(f"[⚠️] Invalid class index {class_idx} at frame {frame_idx}")
                    label = f"Error_{yolo_label}"
                else:
                    label = class_names[class_idx]
                print(f"[Frame {frame_idx}] YOLO: {yolo_label}, Class: {class_idx} → {label}, Confidence: {max_prob:.2f}")

                # Lưu frame và nhãn dự đoán cho feedback
                frame_filename = f"feedback_frame_{frame_idx}_{yolo_label}_{class_idx}.jpg"
                cv2.imwrite(str(feedback_dir / frame_filename), frame)
                with open(feedback_pending_file, "a") as f:
                    f.write(f"{frame_filename},{label}\n")

            except Exception as e:
                print(f"[❌] Classification error at frame {frame_idx}: {e}")
                label = f"Error_{int(cls.item())}"

            # Draw box and label
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1_full, y1_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Done! Output saved to: {output_path}")

# --- Run ---
if __name__ == '__main__':
    video_path = 'Dataset/example2.mp4'
    output_path = 'output_result.mp4'
    detect_and_classify(video_path, output_path)