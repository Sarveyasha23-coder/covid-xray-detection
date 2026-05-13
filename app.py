import streamlit as st
import numpy as np
import cv2
from PIL import Image
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
# Load Model
# -----------------------------------

model = load_model(model_path)

# -----------------------------------
# Show Model Input Shape
# -----------------------------------

st.write("Model Expected Input Shape:", model.input_shape)

# -----------------------------------
# Extract Model Dimensions
# -----------------------------------

input_shape = model.input_shape

height = input_shape[1]
width = input_shape[2]
channels = input_shape[3]

# -----------------------------------
# App Title
# -----------------------------------

st.title("🩺 COVID-19 Detection from Chest X-rays")

st.write(
    "Upload a chest X-ray image to predict whether the patient is COVID Positive or Normal."
)

# -----------------------------------
# Upload Image
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

    # Display image
    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    # -----------------------------------
    # Preprocessing
    # -----------------------------------

    if channels == 1:

        # Convert to grayscale
        image = image.convert("L")

        img = np.array(image)

        # Resize
        img = cv2.resize(img, (width, height))

        # Normalize
        img = img / 255.0

        # Reshape
        img = np.reshape(img, (1, height, width, 1))

    else:

        # Convert to RGB
        image = image.convert("RGB")

        img = np.array(image)

        # Resize
        img = cv2.resize(img, (width, height))

        # Normalize
        img = img / 255.0

        # Reshape
        img = np.reshape(img, (1, height, width, channels))

    # -----------------------------------
    # Debugging Shapes
    # -----------------------------------

    st.write("Image Shape Sent To Model:", img.shape)
    st.write("Model Expected Shape:", model.input_shape)

    # -----------------------------------
    # Predict
    # -----------------------------------

    prediction = model.predict(img)

    # -----------------------------------
    # Results
    # -----------------------------------

    st.subheader("Prediction Result")

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
st.markdown(
    """
    <div style='text-align: center; font-size: 28px; font-weight: 600;'>
        Built by Sarveyasha Sodhiya
    </div>
    """,
    unsafe_allow_html=True
)
