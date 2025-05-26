# Vehicle-Detection-And-Counting
A vehicle detection and counting system using YOLOv11 and Ultralytics' ObjectCounter solution. This project includes a Streamlit web app for uploading and analyzing video/image input, along with standalone Python scripts for video and image processing.

## 🚀 Features

- Real-time vehicle detection and counting using YOLOv11
- Streamlit web interface for user-friendly interaction
- Support for both video and image inputs
- Standalone scripts for batch processing of media files
- Visual representation of detection and counting results

## 📁 Project Structure

```
Vehicle-Detection-And-Counting/
├── assets/                 # Directory for storing assets like images or icons
├── app.py                  # Streamlit web application script
├── image_counting.py       # Script for processing and counting vehicles in images
├── video_counting.py       # Script for processing and counting vehicles in videos
├── requirements.txt        # Python dependencies
├── short_clip.mp4          # Sample video for testing
├── yolo11n.pt              # Pre-trained YOLOv11 model weights
└── README.md               # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Badaszz/Vehicle-Detection-And-Counting.git
   cd Vehicle-Detection-And-Counting
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Streamlit Web Application

Launch the web application using Streamlit:
```bash
streamlit run app.py
```

This will open a web interface where you can upload images or videos for vehicle detection and counting.

### Command-Line Scripts

- **Process an image:**
  ```bash
  python image_counting.py --image_path path_to_image.jpg
  ```

- **Process a video:**
  ```bash
  python video_counting.py --video_path path_to_video.mp4
  ```

Replace `path_to_image.jpg` and `path_to_video.mp4` with the actual paths to your media files.

## 🧠 Model Details

The project employs YOLOv11, a state-of-the-art object detection model, in conjunction with Ultralytics' ObjectCounter for accurate vehicle detection and counting. The pre-trained model weights (`yolo11n.pt`) are included in the repository.

## 📸 Sample Output

Sample outputs can be found in the assets folder

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.

## 📬 Contact

For any inquiries or feedback, please contact [ysolomon298@gmail.com].
