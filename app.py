import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Footwear Image Classifier",
    layout="centered"
)

st.title("👟 Footwear Image Classifier")
st.write("Upload a footwear image and get prediction with confidence.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("real_footwear_model.keras")

model = load_model()

# MUST match training folder order
class_names = ["Boot 👢", "Sandal 🩴", "Shoe 👟"]

# -----------------------------
# IMAGE PREPROCESS
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img = preprocess_image(image)
    predictions = model.predict(img)

    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    st.subheader("Prediction")
    st.success(f"{predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")
