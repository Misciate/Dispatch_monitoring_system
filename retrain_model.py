import os
import shutil
import pandas as pd
from pathlib import Path

# Cấu hình
FEEDBACK_CSV = "feedback.csv"
DATASET_DIR = Path("Dataset/classification_flat")  # Giả sử cấu trúc phẳng

def retrain_with_feedback():
    if not os.path.exists(FEEDBACK_CSV):
        print("No feedback data available.")
        return

    # Đọc feedback
    feedback_data = pd.read_csv(FEEDBACK_CSV)
    if feedback_data.empty:
        print("No new feedback to process.")
        return

    # Di chuyển dữ liệu feedback vào dataset
    for _, row in feedback_data.iterrows():
        src_path = Path(row["frame_path"])
        dest_dir = DATASET_DIR / row["correct_label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        shutil.move(str(src_path), str(dest_path))

    # Gọi script huấn luyện (giả sử train_classification.py có tham số dataset)
    os.system("python train_classification.py --dataset Dataset/classification_flat")

    # Xóa feedback đã xử lý (tùy chọn)
    feedback_data.to_csv(FEEDBACK_CSV, mode='w', index=False)
    print("Model retrained with feedback!")

if __name__ == "__main__":
    retrain_with_feedback()