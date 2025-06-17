import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import torch.nn as nn
from pathlib import Path
import warnings

# --- Load YOLOv5 model ---
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='Models/yolov5/runs/train/dispatch_yolo/weights/best.pt')
yolo_model.conf = 0.25  # Confidence threshold
yolo_model.iou = 0.45   # IoU threshold
yolo_model.classes = None  # Detect both dish and tray
warnings.filterwarnings("ignore", category=FutureWarning)
# --- Load MobileNetV2 classification model ---
mobilenet_model = models.mobilenet_v2(weights=None)
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, 6)  # 6 lớp: 0-5
mobilenet_model.load_state_dict(torch.load('Models/mobilenetv2_classification.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
mobilenet_model.eval()

# --- Class mapping based on YOLO detection ---
class_mapping = {
    0: {0: 'dish/empty', 1: 'dish/kakigori', 2: 'dish/not_empty'},  # Dish (YOLO class 0)
    1: {3: 'tray/empty', 4: 'tray/kakigori', 5: 'tray/not_empty'}   # Tray (YOLO class 1)
}

# --- Preprocessing for classification ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Main function to test model ---
def test_combined_model(input_path, output_path):
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Cannot load image {input_path}")
        return

    # Perform detection with YOLOv5
    results = yolo_model(img)
    boxes = results.xyxy[0]  # Get bounding boxes (x1, y1, x2, y2, conf, cls)

    # Process each detected object
    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        label_det = int(cls.item())  # 0 for dish, 1 for tray
        confidence = conf.item()
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            # Convert to PIL image for classification
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img_pil).unsqueeze(0)
            output = mobilenet_model(input_tensor)
            class_idx = torch.argmax(output, 1).item()  # 0, 1, 2, 3, 4, or 5
            # Map class_idx based on YOLO label
            label = next((v for k, v in class_mapping[label_det].items() if k == class_idx), f"Error_{label_det}")
        except Exception as e:
            print(f"Error in classification: {e}")
            label = f"Error_{label_det}"

        # Draw bounding box and classification label
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save output image
    cv2.imwrite(output_path, img)
    print(f"✅ Output saved to: {output_path}")

# --- Main execution ---
if __name__ == '__main__':
    input_path = 'Dataset\\Detection\\val\\images\\img_001254.jpg'  # Thay bằng đường dẫn ảnh của bạn
    output_path = 'output_detected_image.jpg'
    test_combined_model(input_path, output_path)