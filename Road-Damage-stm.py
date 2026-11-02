import streamlit as st
import json
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="centered"
)

# Title
st.title("🚧 AI Road Damage Detection")
st.write("Upload a road image to detect road damage using AI.")

# Load Class Names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("CAT.keras")

model = load_model()

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Road Image",
             use_container_width=True)

    if st.button("Detect Road Damage"):

        # Image Preprocessing
        img = img.resize((64, 64))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)

        # Prediction
        prediction = model.predict(x)
        predicted_class = class_names[np.argmax(prediction)]

        # Result
        st.success("Prediction Completed!")

        st.subheader("Detection Result")
        st.write("**Predicted Class:**", predicted_class)

        st.write(
            "**Confidence:**",
            f"{np.max(prediction) * 100:.2f}%"
        )
