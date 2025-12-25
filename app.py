import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Footwear Image Classifier",
    page_icon="👟",
    layout="centered"
)

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("real_footwear_model.keras")


model = load_model()

# IMPORTANT: Order must match training class_indices
class_names = ["Boot 👢", "Sandal 🩴", "Shoe 👟"]

# -----------------------------
# UI
# -----------------------------
st.title("👟 Footwear Image Classifier")
st.write(
    "Upload a **real footwear image** and the model will predict whether it is "
    "**Boot, Sandal, or Shoe**."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))

        # Show image (UPDATED: no deprecated params)
        st.image(image, caption="Uploaded Image", width=400)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]

        # Show result
        st.success(
            f"Prediction: **{class_names[predicted_index]}**\n\n"
            f"Confidence: **{confidence:.2f}**"
        )

        # Show confidence scores
        st.subheader("Prediction Confidence")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {prob:.2f}")

    except Exception as e:
        st.error("❌ Error while processing the image.")
        st.exception(e)
