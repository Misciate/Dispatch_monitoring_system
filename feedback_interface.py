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
OUTPUT_VIDEO = "Result_video_[4minutes]"  # Đảm bảo file đã chuyển đổi
FEEDBACK_PENDING = "feedback_pending.txt"

# Khởi tạo file CSV nếu chưa tồn tại
if not os.path.exists(FEEDBACK_CSV):
    pd.DataFrame(columns=["frame_path", "predicted_label", "correct_label", "feedback_time"]).to_csv(FEEDBACK_CSV, index=False)

# Đọc pending feedback (nếu có)
pending_feedback = {}
if os.path.exists(FEEDBACK_PENDING):
    with open(FEEDBACK_PENDING, "r") as f:
        for line in f:
            frame_path, predicted_label = line.strip().split(",", 1)
            pending_feedback[frame_path] = predicted_label

st.title("Dispatch Monitoring Feedback System")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (ICT)")

# Hiển thị video đã detect
if os.path.exists(OUTPUT_VIDEO):
    st.header("Detected Video")
    try:
        video_file = open(OUTPUT_VIDEO, 'rb')
        video_bytes = video_file.read()
        video_file.seek(0)  # Đặt lại con trỏ
        cap = cv2.VideoCapture(OUTPUT_VIDEO)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps  # Thời lượng video (giây)
        cap.release()

        # Slider điều chỉnh thời gian video
        time_seconds = st.slider("Select Time (seconds)", 0, int(duration), 0)
        frame_idx = int((time_seconds * fps) % frame_count)  # Chuyển thời gian sang frame

        # Hiển thị video với điều khiển thời gian
        st.video(video_bytes, format="video/mp4", start_time=time_seconds)

        # Nút Extract and Feedback
        if "extracted_frame" not in st.session_state:
            st.session_state.extracted_frame = None
            st.session_state.predicted_label = None
            st.session_state.correct_label = None

        if st.button("Extract and Feedback"):
            cap = cv2.VideoCapture(OUTPUT_VIDEO)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                st.session_state.extracted_frame = frame
                st.session_state.predicted_label = pending_feedback.get(f"feedback_frame_{frame_idx}.jpg", "Unknown")
                st.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_container_width=True)
                st.write(f"Predicted Label: {st.session_state.predicted_label}")
                st.session_state.correct_label = st.selectbox("Correct Label", [
                    "dish/empty", "dish/kakigori", "dish/not_empty",
                    "tray/empty", "tray/kakigori", "tray/not_empty"
                ], key="correct_label_select")
            else:
                st.error(f"Failed to extract frame at index {frame_idx}. Check if the video frame is accessible.")
            cap.release()

        # Nút Submit Feedback
        if st.session_state.extracted_frame is not None and st.button("Submit Feedback"):
            try:
                frame = st.session_state.extracted_frame
                frame_filename = f"feedback_frame_{frame_idx}.jpg"
                frame_path = FEEDBACK_DIR / frame_filename
                cv2.imwrite(str(frame_path), frame)

                correct_label = st.session_state.correct_label
                confirmed_dir = FEEDBACK_CONFIRMED / correct_label
                confirmed_dir.mkdir(parents=True, exist_ok=True)
                new_frame_path = confirmed_dir / frame_filename
                shutil.move(str(frame_path), str(new_frame_path))

                feedback_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                feedback_data = pd.DataFrame({
                    "frame_path": [str(new_frame_path)],
                    "predicted_label": [st.session_state.predicted_label],
                    "correct_label": [correct_label],
                    "feedback_time": [feedback_time]
                })
                feedback_data.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)

                st.success(f"Feedback for frame {frame_idx} saved at {feedback_time}!")
                st.session_state.extracted_frame = None  # Reset state
                st.rerun()  # Sử dụng st.rerun() thay vì st.experimental_rerun()
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")

    except Exception as e:
        st.error(f"Error loading video or extracting frame: {str(e)}")
else:
    st.warning("No detected video (output_result_converted.mp4) found. Convert output_result.mp4 first.")

# Hiển thị thông tin pending feedback (nếu có)
if pending_feedback:
    st.header("Pending Feedback")
    st.write(f"Total pending frames: {len(pending_feedback)}")
    selected_pending = st.selectbox("Review Pending Frame", list(pending_feedback.keys()))
    if selected_pending:
        frame_path = FEEDBACK_DIR / selected_pending
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
            st.image(frame, channels="BGR", caption=f"Pending Frame: {selected_pending}", use_container_width=True)
            predicted_label = pending_feedback[selected_pending]
            st.write(f"Predicted Label: {predicted_label}")
            correct_label = st.selectbox("Correct Label", [
                "dish/empty", "dish/kakigori", "dish/not_empty",
                "tray/empty", "tray/kakigori", "tray/not_empty"
            ], key=f"correct_label_{selected_pending}")
            if st.button(f"Submit Feedback for {selected_pending}"):
                try:
                    confirmed_dir = FEEDBACK_CONFIRMED / correct_label
                    confirmed_dir.mkdir(parents=True, exist_ok=True)
                    new_frame_path = confirmed_dir / selected_pending
                    shutil.move(str(frame_path), str(new_frame_path))

                    feedback_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feedback_data = pd.DataFrame({
                        "frame_path": [str(new_frame_path)],
                        "predicted_label": [predicted_label],
                        "correct_label": [correct_label],
                        "feedback_time": [feedback_time]
                    })
                    feedback_data.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)

                    with open(FEEDBACK_PENDING, "r") as f:
                        lines = f.readlines()
                    with open(FEEDBACK_PENDING, "w") as f:
                        f.writelines([line for line in lines if not line.startswith(selected_pending)])
                    st.success(f"Feedback for {selected_pending} saved at {feedback_time}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error submitting pending feedback: {str(e)}")
        else:
            st.error(f"Frame {selected_pending} not found!")
else:
    st.write("No pending feedback available.")