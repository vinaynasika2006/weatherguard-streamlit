# 🌦️ WeatherGuardNet

WeatherGuardNet is a computer vision application designed to improve object detection performance under challenging weather conditions such as rain, fog, snow, and low-light environments.

The system enhances visibility and detects objects using a deep learning model integrated into an interactive web interface.

---

## 🚀 Overview

Object detection models often fail in real-world conditions where visibility is poor. WeatherGuardNet addresses this issue by focusing on robust detection using a trained YOLO model and providing an easy-to-use interface for testing and visualization.

The application allows users to upload images and view detected objects directly in the browser.

---

## ✨ Features

* Object detection using YOLOv8
* Handles low-visibility conditions (fog, rain, night)
* Simple and interactive Streamlit interface
* Fast inference with pre-trained model
* Easy to run and test

---

## 🏗️ Project Structure

```
weatherguardnet/
│
├── app.py             # Streamlit application
├── yolov8n.pt         # Pre-trained YOLO model
├── requirements.txt   # Dependencies
├── runtime.txt        # Runtime configuration
├── README.md          # Documentation
```

---

## ⚙️ Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python -m streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 🧪 Usage

1. Launch the application
2. Upload an image
3. The model will process the image
4. Detected objects will be displayed with bounding boxes

---

## 🛠️ Technologies Used

* Python
* Streamlit
* PyTorch
* YOLOv8 (Ultralytics)

---

## ⚠️ Notes

* Make sure `yolov8n.pt` is present in the project folder
* First run may take slightly longer due to model loading
* Works best with Python 3.10 or 3.11

---

## 🔮 Future Improvements

* Add weather condition detection module
* Integrate image enhancement for better visibility
* Support real-time video processing
* Improve detection accuracy in extreme conditions
* Deploy as a cloud-based application
