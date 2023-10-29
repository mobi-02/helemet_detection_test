import streamlit as st
from PIL import Image, ImageDraw, ImageFont  # Add ImageDraw here
from ultralytics import YOLO
import numpy as np
import cv2

st.title('Image Processing with Streamlit')

model = YOLO("best.pt")

def evaluate(results, frame):
    boxes_data = results[0].boxes
    processed_frame = frame.copy()
    pil_frame = Image.fromarray(processed_frame)
    draw = ImageDraw.Draw(pil_frame)

    encoder = {0: "Helmet", 1: "No Helmet"}

    for class_, conf, box in zip(boxes_data.cls, boxes_data.conf, boxes_data.xyxy):
        if conf > 0.1:
            points = [int(point) for point in box]
            x1, y1, x2, y2 = points
            color = "green" if int(class_) == 0 else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

            label = f"Class: {encoder[int(class_)]}, Confidence: {conf:.2f}"
            draw.text((x1 + 5, y1 + 10), label, fill=color)

    processed_frame = np.array(pil_frame)
    return processed_frame

def process_image(image_path):
    frame = cv2.imread(image_path)
    results = model(frame, conf=0.01)
    processed_frame = evaluate(results, frame)

    return processed_frame

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image_path = "uploaded_image.jpg"  # You can use a temporary file to save the uploaded image
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())

    processed_image = process_image(image_path)
    st.image(processed_image, caption="Processed Image", use_column_width=True)
