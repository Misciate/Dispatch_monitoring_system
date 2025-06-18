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
OUTPUT_VIDEO = "Dataset/Result_video_[4minutes].mp4"
FEEDBACK_PENDING = "feedback_pending.txt"

# Khởi tạo file CSV nếu chưa tồn tại
if not os.path.exists(FEEDBACK_CSV):
    pd.DataFrame(columns=["frame_path", "predicted_label", "correct_label", "feedback_time"]).to_csv(FEEDBACK_CSV, index=False)

# Đọc pending feedback (không xử lý chi tiết file trong feedback_pending theo yêu cầu)
pending_feedback = {}
pending_frame_info = []

if os.path.exists(FEEDBACK_PENDING):
    with open(FEEDBACK_PENDING, "r") as f:
        for line in f:
            frame_path, predicted_label = line.strip().split(",", 1)
            parts = frame_path.replace("feedback_frame_", "").replace(".jpg", "").split("_")
            try:
                frame_idx = int(parts[-1])
                pending_feedback[frame_path] = {
                    "predicted_label": predicted_label,
                    "frame_idx": frame_idx
                }
                pending_frame_info.append((frame_idx, frame_path))
            except:
                continue

# Giao diện
st.title("Dispatch Monitoring Feedback System")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (ICT)")

if os.path.exists(OUTPUT_VIDEO):
    st.header("Detected Video")
    try:
        video_file = open(OUTPUT_VIDEO, 'rb')
        video_bytes = video_file.read()

        cap = cv2.VideoCapture(OUTPUT_VIDEO)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = float(frame_count) / fps
        cap.release()

        if pending_frame_info:
            time_seconds = st.slider("Select Time (seconds)", 0.0, duration, 0.0, step=1.0/fps, format="{:.2f}s")
            current_frame = int((time_seconds * fps) % frame_count)
            nearest_pending = min(pending_frame_info, key=lambda x: abs(x[0] - current_frame))
            frame_idx = nearest_pending[0]
            frame_filename = nearest_pending[1]
            predicted_label = pending_feedback[frame_filename]["predicted_label"]
        else:
            st.warning("No pending feedback frames available.")
            st.stop()

        st.video(video_bytes, format="video/mp4", start_time=float(time_seconds))

        # Khởi tạo hoặc cập nhật trạng thái
        if "extracted_frame" not in st.session_state:
            st.session_state.extracted_frame = None
            st.session_state.predicted_label = None
            st.session_state.correct_label = None
            st.session_state.current_feedback_frame = None

        if st.button("Extract and Feedback"):
            cap = cv2.VideoCapture(OUTPUT_VIDEO)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                st.session_state.extracted_frame = frame
                st.session_state.predicted_label = predicted_label
                st.session_state.current_feedback_frame = frame_filename
                st.image(frame, channels="BGR", caption=frame_filename, use_container_width=True)
                st.write(f"Predicted Label: {predicted_label}")
                st.success(f"Successfully extracted frame {frame_idx}")
                # Cập nhật correct_label ngay khi chọn từ selectbox
                if "correct_label_select" not in st.session_state:
                    st.session_state.correct_label = None
                st.session_state.correct_label = st.selectbox("Correct Label", [
                    "dish/empty", "dish/kakigori", "dish/not_empty",
                    "tray/empty", "tray/kakigori", "tray/not_empty"
                ], index=0 if st.session_state.correct_label is None else [
                    "dish/empty", "dish/kakigori", "dish/not_empty",
                    "tray/empty", "tray/kakigori", "tray/not_empty"
                ].index(st.session_state.correct_label) if st.session_state.correct_label in [
                    "dish/empty", "dish/kakigori", "dish/not_empty",
                    "tray/empty", "tray/kakigori", "tray/not_empty"
                ] else 0, key="correct_label_select")
            else:
                st.error(f"Could not extract frame at index {frame_idx}")

        # Nút Submit Feedback
        if st.session_state.extracted_frame is not None and st.button("Submit Feedback"):
            if st.session_state.correct_label is None:
                st.error("Please select a Correct Label before submitting.")
            else:
                try:
                    frame = st.session_state.extracted_frame
                    predicted_label = st.session_state.predicted_label
                    correct_label = st.session_state.correct_label
                    frame_filename = st.session_state.current_feedback_frame

                    frame_path = FEEDBACK_DIR / frame_filename
                    cv2.imwrite(str(frame_path), frame)

                    confirmed_dir = FEEDBACK_CONFIRMED / correct_label
                    confirmed_dir.mkdir(parents=True, exist_ok=True)
                    new_frame_path = confirmed_dir / frame_filename
                    shutil.move(str(frame_path), str(new_frame_path))

                    feedback_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feedback_data = pd.DataFrame({
                        "frame_path": [str(new_frame_path)],
                        "predicted_label": [predicted_label],
                        "correct_label": [correct_label],
                        "feedback_time": [feedback_time]
                    })
                    feedback_data.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)

                    # Xóa khỏi feedback_pending.txt
                    if frame_filename in pending_feedback:
                        with open(FEEDBACK_PENDING, "r") as f:
                            lines = f.readlines()
                        with open(FEEDBACK_PENDING, "w") as f:
                            f.writelines([line for line in lines if not line.startswith(frame_filename)])
                        del pending_feedback[frame_filename]

                    st.success(f"Feedback for {frame_filename} saved successfully at {feedback_time}")
                    st.session_state.extracted_frame = None
                    st.session_state.correct_label = None  # Reset correct_label sau khi submit
                    st.rerun()
                except Exception as e:
                    st.error(f"Error submitting feedback: {str(e)}")

    except Exception as e:
        st.error(f"Error loading video or extracting frame: {str(e)}")
else:
    st.warning("No detected video (Dataset/Result_video_[4minutes].mp4) found.")