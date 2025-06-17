import cv2
import numpy as np

# --- Cấu hình ---
video_path = 'Dataset/example2.mp4'  # Đường dẫn video
output_size = (400, 200)  # Kích thước vùng crop sau khi "bẻ" (W, H)

# --- Đọc frame đầu tiên ---
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Không thể đọc frame đầu tiên.")
    exit()

# --- Chọn 4 điểm bằng chuột ---
points = []
clone = frame.copy()

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", clone)

print("🖱️ Click chuột trái chọn 4 điểm (theo thứ tự: TL → TR → BR → BL)")
cv2.imshow("Select 4 Points", clone)
cv2.setMouseCallback("Select 4 Points", click_event)

# Đợi đến khi chọn đủ 4 điểm
while True:
    key = cv2.waitKey(1) & 0xFF
    if len(points) == 4:
        break
cv2.destroyAllWindows()

# --- Perspective Transform ---
pts_src = np.array(points, dtype="float32")
w, h = output_size
pts_dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(frame, M, output_size)

# --- Hiển thị kết quả ---
cv2.imshow("Warped Crop (ROI)", warped)
cv2.imwrite("cropped_roi.png", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- In ra tọa độ gốc đã chọn ---
print("✅ Tọa độ 4 điểm đã chọn:")
for i, pt in enumerate(points):
    print(f"Point {i+1}: {pt}")