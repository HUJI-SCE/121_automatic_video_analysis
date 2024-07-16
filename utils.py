import os
import time
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st

VIDEO_DIR = "videos"
RESULTS_DIR = "results"


def display_video_upload_section():
    st.header("Upload your video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    return uploaded_file


def process_video_upload(uploaded_file):
    video_path = os.path.join(VIDEO_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    return video_path


def display_time_inputs(video_length):
    col1, col2 = st.columns(2)
    with col1:
        start_time_input = st.number_input("Start time (seconds)", min_value=0.0, value=0.0, step=1.0)
    with col2:
        end_time_input = st.number_input("End time (seconds)", min_value=0.0, value=video_length, step=1.0)
    return start_time_input, end_time_input


def display_control_buttons():
    col3, col4 = st.columns(2)
    with col3:
        start_button = st.button("Start")
    with col4:
        stop_button = st.button("Stop")

    if "stop" not in st.session_state:
        st.session_state.stop = False

    if stop_button:
        st.session_state.stop = True

    if start_button:
        st.session_state.stop = False

    return start_button, stop_button


def display_frame_results(class_names):
    frames_per_page = 10
    total_frames = len(st.session_state.frames)
    total_pages = (total_frames + frames_per_page - 1) // frames_per_page
    current_page = st.session_state.get("current_page", 0)

    st.write(f"Showing page {current_page + 1} of {total_pages}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous", key="prev_page") and current_page > 0:
            st.session_state.current_page -= 1
    with col2:
        if st.button("Next", key="next_page") and current_page < total_pages - 1:
            st.session_state.current_page += 1

    start_idx = current_page * frames_per_page
    end_idx = min(start_idx + frames_per_page, total_frames)

    for i in range(start_idx, end_idx):
        frame = st.session_state.frames[i]
        validation = st.session_state.validations[i]

        st.image(frame, caption=f"Frame {i + 1}", use_column_width=True)
        st.write(f"Predicted Label: {validation['prediction']}")

        correct_class = st.selectbox("Correct Label", class_names,
                                     index=class_names.index(validation['correct_class']),
                                     key=f"class_{i}")

        st.session_state.validations[i]['correct_class'] = correct_class

    if st.button("Save Results"):
        results = st.session_state.validations
        df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, "validation_results.csv")
        df.to_csv(csv_path, index=False)
        st.write(f"Results saved to {csv_path}")


def predict_frame(frame, model, knn, transform, class_names):
    frame = transform(frame).unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(frame, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-4:]
        features = torch.cat([hidden_state[:, 0, :] for hidden_state in last_hidden_states], dim=-1)
    features = features.cpu().numpy()
    prediction = knn.predict(features)[0]
    return class_names[prediction]


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"


def process_video(video_path, model, knn, transform, start_time, end_time, class_names):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    predictions = []
    time_intervals = []

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    if end_frame > frame_count:
        end_frame = frame_count

    st.write(f"Number of frames in the video: {frame_count}")
    progress_bar = st.progress(0)
    frame_processing_time = []
    time_placeholder = st.empty()  # Placeholder for the estimated time
    frame_placeholder = st.empty()  # Placeholder for the current frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    processing_start_time = time.time()
    while current_frame < end_frame:
        if st.session_state.get("stop", False):
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        start_frame_time = time.time()
        label = predict_frame(frame_pil, model, knn, transform, class_names)
        end_frame_time = time.time()
        frame_processing_time.append(end_frame_time - start_frame_time)

        frames.append(frame_pil)
        predictions.append(label)

        progress = (current_frame - start_frame + 1) / (end_frame - start_frame)
        progress_bar.progress(progress)

        elapsed_time = time.time() - processing_start_time
        avg_frame_time = np.mean(frame_processing_time)
        remaining_time = avg_frame_time * (end_frame - current_frame - 1)

        formatted_time = format_time(remaining_time)
        time_placeholder.write(
            f"Number of frames: {end_frame - start_frame} | Estimated time until finish: {formatted_time}")
        frame_placeholder.write(f"Processing frame {current_frame} of {end_frame}")

        # Calculate the time interval for the current frame
        current_time = current_frame / fps
        if len(time_intervals) == 0 or time_intervals[-1]['label'] != label:
            if len(time_intervals) > 0:
                time_intervals[-1]['end'] = current_time
            time_intervals.append({'label': label, 'start': current_time, 'end': current_time})

        current_frame += 1

    if len(time_intervals) > 0:
        time_intervals[-1]['end'] = end_frame / fps

    cap.release()
    return frames, predictions, time_intervals


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = frame_count / fps
    cap.release()
    return video_length
