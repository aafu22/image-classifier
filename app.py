import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Footwear Image Classifier",
    layout="centered"
)

st.title("👟 Footwear Image Classifier")
st.write("Upload an image of footwear and get predictions")

# ---------------------------------
# LOAD MODEL (Keras 3 safe)
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("real_footwear_model.keras")

model = load_model()

# IMPORTANT: Must match training folder order
class_names = ["Boot 👢", "Sandal 🩴", "Shoe 👟"]

# ---------------------------------
# IMAGE PREPROCESSING
# ---------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")          # Ensure 3 channels
    image = image.resize((224, 224))      # MobileNet input size
    img_array = np.array(image) / 255.0   # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------------
# FILE UPLOAD
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload footwear image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    predicted_class = class_names[predicted_index]

    # ---------------------------------
    # RESULTS
    # ---------------------------------
    st.markdown("### 🧠 Prediction Result")
    st.success(f"**{predicted_class}**")
    st.info(f"Prediction Confidence: **{confidence:.2f}%**")

    # Show all class probabilities
    st.markdown("### 📊 Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")

else:
    st.warning("Please upload an image to start prediction.")
