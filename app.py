import streamlit as st

# Set the page configuration
st.set_page_config(page_title="Vehicle Detection and Counting", layout="centered") 

# Trying to import cv2

import sys
import subprocess

# Check if OpenCV is available
try:
    import cv2
    st.write("OpenCV version:", cv2.__version__)
except ImportError:
    st.write("cv2 not found, trying to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"])
    import cv2
    st.write("OpenCV installed and imported successfully!")



import tempfile
import numpy as np
from ultralytics import solutions
from PIL import Image

# Title and description
st.title("🚗 Vehicle Detection and Counting")
st.write("Upload an image or video to detect and count vehicles using the YOLOv11 model from ultralytics.")

# choice to select input type
option = st.radio("Select input type:", ("Image", "Video"))

# Setup ObjectCounter
counter = solutions.ObjectCounter(
    show=False,
    model="yolo11n.pt",
    classes=[2],  # class 2 = cars # class 0 for persons
    tracker="botsort.yaml"
)

#function to process image
def process_image(img: np.ndarray):
    # process the image using the object counter
    results = counter.process(img)
    # get the track IDs of detected vehicles
    track_ids = counter.track_ids
    # set to store unique vehicle IDs
    unique_vehicle_ids = set(int(id) for id in track_ids) if track_ids else set()
    # get the total count of unique vehicles
    total_count = len(unique_vehicle_ids)
    # draw the total count on the image
    cv2.putText(results.plot_im, f"Detected: {total_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # return the processed image and total count
    return results.plot_im, total_count

def process_video(video_path, output_path="processed_output.mp4"):
    cap = cv2.VideoCapture(video_path)
    unique_vehicle_ids = set()

    # Read first frame to get correct dimensions
    success, frame = cap.read()
    if not success:
        st.error("Failed to read the video file.")
        return None

    results = counter.process(frame)
    height, width, _ = results.plot_im.shape
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback if fps is 0

    # Re-initialize capture since we already read the first frame
    cap.release()
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = counter.process(frame)
        track_ids = counter.track_ids

        if results and track_ids is not None:
            for track_id in track_ids:
                unique_vehicle_ids.add(int(track_id))

        current_count = len(track_ids) if track_ids else 0
        total_count = len(unique_vehicle_ids)

        cv2.putText(results.plot_im, f"Detected: {current_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(results.plot_im, f"Total: {total_count}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(results.plot_im)

    cap.release()
    out.release()
    return output_path


if option == "Image":
    # Upload an image file
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Check if an image is uploaded
    if uploaded_img is not None:
        # preprocess the image
        image = Image.open(uploaded_img).convert("RGB")
        image_np = np.array(image)
        
        #process the image using the object counter
        output_img, count = process_image(image_np)

        # Display the processed image with the count
        st.image(output_img, caption=f"Vehicles Detected: {count}", channels="BGR", use_container_width=True)

elif option == "Video":
    # Upload a video file
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    # Check if a video is uploaded
    if uploaded_vid is not None:
        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        # Save the uploaded video to the temporary file (which is just a bunch of frames))
        tfile.write(uploaded_vid.read())
        stframe = st.empty()
        
        # Display the video in the Streamlit app frame by frame
        st.write("Processing video...")
        processed_video_path = process_video(tfile.name)

        # Show the video with Streamlit's native video player
        with open(processed_video_path, 'rb') as video_file:
            st.video(video_file.read())

