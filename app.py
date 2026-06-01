import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ Solar Panel Defect Classifier")
st.write("Upload a solar panel image to detect defects.")

# -----------------------------
# CLASS NAMES
# -----------------------------
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            r"C:\Users\adder\OneDrive\Documents\Revanth\Deep Leaning Project\Data\trained_effnet_finetuned.h5",
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model:\n{e}")
        return None

with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Solar Panel Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # PREPROCESS IMAGE
    # -----------------------------
    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(
        img_array.astype(np.float32)
    )

    # -----------------------------
    # PREDICTION
    # -----------------------------
    with st.spinner("Analyzing Solar Panel..."):

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_idx = np.argmax(predictions[0])

        confidence = predictions[0][predicted_idx]

        predicted_class = CLASSES[predicted_idx]

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2%}"
    )

    if predicted_class == "Clean":
        st.success(
            "The solar panel appears to be in good condition."
        )
    else:
        st.warning(
            "Defect or contamination detected."
        )

    # -----------------------------
    # TOP 3 PREDICTIONS
    # -----------------------------
    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(
        predictions[0]
    )[-3:][::-1]

    for rank, idx in enumerate(top_indices, start=1):

        class_name = CLASSES[idx]

        probability = predictions[0][idx]

        st.write(
            f"{rank}. {class_name} : {probability:.2%}"
        )

    # -----------------------------
    # ALL PROBABILITIES
    # -----------------------------
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(CLASSES):

        prob = float(predictions[0][i])

        st.write(
            f"{class_name} : {prob:.2%}"
        )

        st.progress(prob)