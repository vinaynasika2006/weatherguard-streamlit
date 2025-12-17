import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import tempfile

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="WeatherGuard YOLO", layout="wide")
st.title("WeatherGuard – Adaptive Video Object Detection")

# ----------------------------
# Device info
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device.upper()}")

# ----------------------------
# Load model once
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------------
# Image restoration
# ----------------------------
def anti_glare_restoration(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
    img = np.minimum(img, 0.95) / 0.95
    img = np.power(img, 0.9)
    img = (img * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.bilateralFilter(img, 7, 70, 70)
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)

    return img

# ----------------------------
# Adaptation engine
# ----------------------------
def adaptation_engine(env):
    if env == "clear":
        return 0.7, 50
    if env == "rainy":
        return 0.4, 30
    if env == "foggy":
        return 0.35, 70
    if env == "glare":
        return 0.6, 100
    return 0.25, 0

# ----------------------------
# Detection
# ----------------------------
def detect(frame, conf, min_area):
    results = model(frame, conf=conf, verbose=False)
    out = frame.copy()

    if results and results[0].boxes:
        for x1, y1, x2, y2, score, cls in results[0].boxes.data.tolist():
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = model.names[int(cls)]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{label} {score:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
    return out

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")
environment = st.sidebar.selectbox(
    "Environment",
    ["clear", "rainy", "foggy", "glare"]
)
start = st.sidebar.button("Start Processing")

# ----------------------------
# Upload
# ----------------------------
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# ----------------------------
# Processing
# ----------------------------
if video_file and start:
    conf, min_area = adaptation_engine(environment)

    st.write(f"Confidence: {conf}")
    st.write(f"Min bbox area: {min_area}")

    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    temp_in.write(video_file.read())
    temp_in.close()

    cap = cv2.VideoCapture(temp_in.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        temp_out.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        restored = anti_glare_restoration(pil)
        detected = detect(restored, conf, min_area)

        writer.write(detected)
        frame_box.image(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
    writer.release()

    st.success("Processing complete")

    with open(temp_out.name, "rb") as f:
        st.download_button(
            "Download processed video",
            f,
            file_name="weatherguard_output.mp4",
            mime="video/mp4"
        )
