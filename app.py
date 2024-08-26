import os
import pickle
import pandas as pd
import torch
import yaml
from torchvision import transforms
from transformers import ViTForImageClassification
from xgboost.testing import datasets
from xgboost.testing.data import joblib

from training import PipeTrainer
from utils import process_video, get_video_length
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

class_names = ['bubble_air', 'welding', 'faucet', 'clean']
VIDEO_DIR = "./videos"
RESULTS_DIR = "./results"
MODEL_PATH = "./models"
CONFIG_PATH = "config.yaml"


def ensure_directories():
    """
    Ensures that the required directories for storing videos, results, and models exist. 
    Creates them if they do not already exist.
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)


def load_config():
    """
    Loads the configuration settings from a YAML file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(CONFIG_PATH, "r") as config_file:
        return yaml.safe_load(config_file)


def load_models(config):
    """
    Loads the Vision Transformer (ViT) model and the k-Nearest Neighbors (k-NN) classifier 
    from the specified paths in the configuration.

    Args:
        config (dict): The configuration dictionary containing model paths.

    Returns:
        tuple: A tuple containing the ViT model and the k-NN classifier.
    """
    model = ViTForImageClassification.from_pretrained(config['model_name'])
    model.eval()
    with open(config['knn_model_path'], "rb") as f:
        knn = joblib.load(f)
    return model, knn


def display_video_upload_section():
    """
    Displays a section in the Streamlit interface for uploading a video file.

    Returns:
        UploadedFile: The file object uploaded by the user.
    """
    st.header("Upload your video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    return uploaded_file


def process_video_upload(uploaded_file):
    """
    Saves the uploaded video file to the specified directory.

    Args:
        uploaded_file (UploadedFile): The uploaded video file.

    Returns:
        str: The file path where the video is saved.
    """
    video_path = os.path.join(VIDEO_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    return video_path


def display_time_inputs(video_length):
    """
    Displays input fields in the Streamlit interface for selecting the start and end times of the video processing.

    Args:
        video_length (float): The total length of the video in seconds.

    Returns:
        tuple: A tuple containing the start time and end time inputs as floats.
    """
    col1, col2 = st.columns(2)
    with col1:
        start_time_input = st.number_input("Start time (seconds)", min_value=0.0, value=0.0, step=1.0)
    with col2:
        end_time_input = st.number_input("End time (seconds)", min_value=0.0, value=video_length, step=1.0)
    return start_time_input, end_time_input


def display_control_buttons():
    """
    Displays control buttons (Start and Stop) in the Streamlit interface for controlling the video processing.

    Returns:
        tuple: A tuple containing the states of the start and stop buttons as booleans.
    """
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
    """
    Displays the processed frames and their predicted labels, allows the user to correct the predictions, 
    and saves the results.

    Args:
        class_names (list): A list of class names for predictions.
    """
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

    for i in range(start_idx, end_idx, frames_per_page):
        frames_batch = st.session_state.frames[i:i + frames_per_page]
        predictions_batch = st.session_state.predictions[i:i + frames_per_page]

        # Display each frame in the batch
        for j, frame in enumerate(frames_batch):
            st.image(frame, caption=f"Frame {i + j + 1}", use_column_width=True)
            st.write(f"Predicted Label: {predictions_batch[j]}")

        # Committee result calculation
        committee_result = max(set(predictions_batch), key=predictions_batch.count)
        st.write(f"Committee Result for Frames {i + 1} to {i + len(frames_batch)}: {committee_result}")

        # Select the correct class for the entire batch
        correct_class = st.selectbox("Correct Label for this batch", class_names,
                                     index=class_names.index(committee_result),
                                     key=f"batch_class_{i // frames_per_page}")

        for j in range(frames_per_page):
            if i + j < total_frames:
                st.session_state.validations[i + j]['correct_class'] = correct_class

    if st.button("Save Results"):
        results = st.session_state.validations
        df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, "validation_results.csv")
        df.to_csv(csv_path, index=False)
        st.write(f"Results saved to {csv_path}")


def main_page(model, knn, transform, class_names):
    """
    The main page for the Streamlit application. Handles video upload, processing, and displaying results.

    Args:
        model (ViTForImageClassification): The Vision Transformer model.
        knn (KNeighborsClassifier): The k-NN classifier.
        transform (transforms.Compose): The image transformation pipeline.
        class_names (list): A list of class names for predictions.
    """
    st.title("Automatic Video Analysis")
    uploaded_file = display_video_upload_section()
    finished=False
    if uploaded_file:
        video_path = process_video_upload(uploaded_file)
        video_length = get_video_length(video_path)
        start_time_input, end_time_input = display_time_inputs(video_length)
        start_button, stop_button = display_control_buttons()

        if start_button:
            frames, predictions, time_intervals = process_video(video_path, model, knn, transform, start_time_input,
                                                                end_time_input, class_names)
            st.session_state.frames = frames
            st.session_state.predictions = predictions
            st.session_state.time_intervals = time_intervals
            st.session_state.validations = [{'frame_number': i + 1, 'prediction': pred, 'correct_class': pred} for
                                            i, pred in enumerate(predictions)]
            st.session_state.current_page = 0
            finished=True

    if "frames" in st.session_state and "validations" in st.session_state and finished:
        display_frame_results(class_names)


def train_new_data_page():
    """
    The page for training new data in the Streamlit application.
    Allows the user to select a directory of images, specify a device, and train a new model.
    """
    st.title("Train New Data")

    # Replace the drag and drop with directory selection
    data_dir = st.text_input("Enter the directory path where the data is located")
    gpu_device = st.selectbox("Select GPU device", ["CUDA", "MPS", "CPU"])

    if st.button("Train"):
        if not os.path.isdir(data_dir):
            st.error("Invalid directory path. Please enter a valid directory.")
            return

        if gpu_device == "CUDA" and not torch.cuda.is_available():
            st.error("CUDA is not available. Please select another device.")
            return
        if gpu_device == "MPS" and not torch.backends.mps.is_available():
            st.error("MPS is not available. Please select another device.")
            return

        device = "cuda" if gpu_device == "CUDA" else "mps" if gpu_device == "MPS" else "cpu"

        st.write(f"Loading data from {data_dir}...")

        # Define a transform to preprocess the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the dataset
        train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        class_names = train_dataset.classes  # Get class names from the dataset
        st.write(f"Found {len(train_dataset)} images belonging to {len(class_names)} classes.")

        # Initialize the trainer
        trainer = PipeTrainer(train_dataset, class_names)

        # Train the model
        st.write("Training started...")
        with st.spinner("Training the model, please wait..."):
            classifiers_concat, classifiers_cls = trainer.train()

        st.success("Training completed. Classifiers saved.")


def settings_page(config):
    """
    The settings page for the Streamlit application.
    Allows the user to view and update the currently used KNN model.

    Args:
        config (dict): The configuration dictionary containing model paths and other settings.
    """
    st.title("Settings")

    st.subheader("Currently Used KNN Model")
    st.write(f"Current KNN model: {config['knn_model_path']}")

    st.subheader("Load a New KNN Model")
    knn_model_file = st.file_uploader("Upload KNN model (.pkl)", type=["pkl"])

    if st.button("Load and Save KNN Model"):
        if knn_model_file is not None:
            new_model_path = os.path.join(MODEL_PATH, knn_model_file.name)
            with open(new_model_path, "wb") as f:
                f.write(knn_model_file.read())
            config['knn_model_path'] = new_model_path
            with open(CONFIG_PATH, "w") as config_file:
                yaml.safe_dump(config, config_file)
            st.success("New KNN model loaded and configuration updated. Please restart the app to apply changes.")

    st.subheader("Available Models in Models Folder")
    available_models = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
    selected_model = st.selectbox("Select a model to use", available_models)

    if st.button("Update Config with Selected Model"):
        selected_model_path = os.path.join(MODEL_PATH, selected_model)
        config['knn_model_path'] = selected_model_path
        with open(CONFIG_PATH, "w") as config_file:
            yaml.safe_dump(config, config_file)
        st.success("Configuration updated with selected model. Please restart the app to apply changes.")


def performance_page():
    """
    Placeholder for the performance page in the Streamlit application.
    """
    st.title("Model Performance")


def last_uses_page():
    """
    Placeholder for the page displaying previous predictions in the Streamlit application.
    """
    st.title("Previous Predictions")


def main():
    """
    The main function for the Streamlit application.
    Handles navigation between different pages and loads necessary models and configurations.
    """
    ensure_directories()
    config = load_config()
    model, knn = load_models(config)

    # Call this function to display the logos
    lst = ["Main Page", 'Train New Data', "Settings"]
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", lst)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if page == lst[0]:
        main_page(model, knn, transform, class_names)
    elif page == lst[1]:
        train_new_data_page()
    elif page == lst[2]:
        settings_page(config)
    # elif page == "Performance":
    #     performance_page()
    # elif page == "Last Uses":
    #     last_uses_page()


if __name__ == '__main__':
    """
    The entry point of the script.
    Calls the `main()` function to start the Streamlit application.
    """
    main()
