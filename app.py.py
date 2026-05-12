
import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model

model = load_model('covid_model.h5')

classes = ['Covid', 'Normal', 'Viral Pneumonia']

st.title("COVID Detection from Chest X-Ray")

uploaded_file = st.file_uploader(
    "Upload X-Ray Image",
    type=['jpg','png','jpeg']
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, 1)

    image = cv2.resize(image, (128,128))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_class = classes[np.argmax(prediction)]

    st.image(uploaded_file)

    st.success(f"Prediction: {predicted_class}")
