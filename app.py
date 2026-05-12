import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import os

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="COVID-19 X-ray Detection",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------------
# Download Model from Google Drive
# -----------------------------------

file_id = "1WOcCDbuGZsGOxKI-ViO9L0JQcjN4myTi"
model_path = "covid_model.h5"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# -----------------------------------
# Load Trained Model
# -----------------------------------

model = load_model(model_path)

# -----------------------------------
# App Title
# -----------------------------------

st.title("🩺 COVID-19 Detection from Chest X-rays")

st.write(
    "Upload a chest X-ray image and the CNN model will predict whether the patient is COVID Positive or Normal."
)

# -----------------------------------
# File Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Prediction Section
# -----------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert to numpy array
    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (224, 224))

    # Normalize image
    img = img / 255.0

    # Reshape image for model
    img = np.reshape(img, (1, 224, 224, 3))

    # -----------------------------------
    # Predict
    # -----------------------------------

    prediction = model.predict(img)

    st.subheader("Prediction Result")

    # Binary Classification
    if prediction[0][0] > 0.5:
        st.error("⚠️ COVID Positive")
        confidence = prediction[0][0] * 100
    else:
        st.success("✅ Normal")
        confidence = (1 - prediction[0][0]) * 100

    st.write(f"Confidence Score: {confidence:.2f}%")

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.markdown(
    "Developed using CNN, TensorFlow, Streamlit, and Chest X-ray Imaging"
)
