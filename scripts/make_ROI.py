import cv2
import numpy as np

video_path = 'Dataset/example_1.mp4'
output_size = (600, 200)  # Output warped size (width, height)
display_width = 960       # Max display width
display_height = 540      # Max display height

# Đọc frame đầu
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Không đọc được frame đầu.")
    exit()

original_h, original_w = frame.shape[:2]

# Tính tỷ lệ scale để hiển thị vừa màn hình
scale_w = display_width / original_w
scale_h = display_height / original_h
scale = min(scale_w, scale_h)

# Resize để hiển thị
resized = cv2.resize(frame, (int(original_w * scale), int(original_h * scale)))
clone = resized.copy()
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((int(x / scale), int(y / scale)))  # Scale ngược lại
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4-Point ROI", clone)

print("👉 Click 4 điểm nghiêng theo thứ tự: TL → TR → BR → BL")
cv2.imshow("Select 4-Point ROI", clone)
cv2.setMouseCallback("Select 4-Point ROI", click_event)

while len(points) < 4:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Perspective transform
pts_src = np.array(points, dtype="float32")
w, h = output_size
pts_dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(frame, M, (w, h))

cv2.imshow("Warped Crop", warped)
cv2.imwrite("warped_cropped_roi.png", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ 4 điểm gốc (trên ảnh gốc):")
for i, pt in enumerate(points):
    print(f"Point {i+1}: {pt}")
np.save("perspective_matrix.npy", M)
