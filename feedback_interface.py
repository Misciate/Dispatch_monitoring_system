import streamlit as st
import cv2
import os
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

# Cấu hình thư mục và file
FEEDBACK_DIR = Path("feedback")
FEEDBACK_CONFIRMED = FEEDBACK_DIR / "confirmed"
FEEDBACK_CSV = "feedback.csv"
FEEDBACK_PENDING = "feedback_pending.txt"

# Khởi tạo file CSV nếu chưa tồn tại
if not os.path.exists(FEEDBACK_CSV):
    pd.DataFrame(columns=["frame_path", "predicted_label", "correct_label", "feedback_time"]).to_csv(FEEDBACK_CSV, index=False)

# Đọc pending feedback
pending_feedback = {}
if os.path.exists(FEEDBACK_PENDING):
    with open(FEEDBACK_PENDING, "r") as f:
        for line in f:
            frame_path, predicted_label = line.strip().split(",", 1)  # Chỉ tách 1 lần để hỗ trợ nhãn có dấu phẩy
            pending_feedback[frame_path] = predicted_label

st.title("Dispatch Monitoring Feedback System")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (ICT)")

# Hiển thị frame từ pending feedback
if not pending_feedback:
    st.warning("No pending feedback available. Run `detect_and_classify.py` first to generate frames.")
else:
    st.write(f"Total pending frames: {len(pending_feedback)}")
    selected_frame = st.selectbox("Select Frame to Review", list(pending_feedback.keys()))
    
    if selected_frame:
        frame_path = FEEDBACK_DIR / selected_frame
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
            st.image(frame, channels="BGR", caption=f"Frame: {selected_frame}", use_column_width=True)

            predicted_label = pending_feedback[selected_frame]
            st.write(f"Predicted Label: {predicted_label}")
            correct_label = st.selectbox("Correct Label", [
                "dish/empty", "dish/kakigori", "dish/not_empty",
                "tray/empty", "tray/kakigori", "tray/not_empty"
            ])

            if st.button("Submit Feedback"):
                # Tạo thư mục confirmed nếu chưa tồn tại
                confirmed_dir = FEEDBACK_CONFIRMED / correct_label
                confirmed_dir.mkdir(parents=True, exist_ok=True)
                
                # Di chuyển frame vào thư mục confirmed với nhãn đúng
                new_frame_path = confirmed_dir / selected_frame
                shutil.move(str(frame_path), str(new_frame_path))

                # Ghi vào CSV với timestamp
                feedback_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                feedback_data = pd.DataFrame({
                    "frame_path": [str(new_frame_path)],
                    "predicted_label": [predicted_label],
                    "correct_label": [correct_label],
                    "feedback_time": [feedback_time]
                })
                feedback_data.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)

                # Xóa khỏi pending sau khi xử lý
                with open(FEEDBACK_PENDING, "r") as f:
                    lines = f.readlines()
                with open(FEEDBACK_PENDING, "w") as f:
                    f.writelines([line for line in lines if not line.startswith(selected_frame.split(",")[0])])

                st.success(f"Feedback for {selected_frame} saved at {feedback_time}!")
        else:
            st.error(f"Frame {selected_frame} not found! It may have been processed already.")