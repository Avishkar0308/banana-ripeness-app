import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

# Set page config
st.set_page_config(page_title="Banana Ripeness Predictor 🍌", layout="centered")

# Custom CSS for pro styling
st.markdown("""
    <style>
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #FACC15;
    }
    .subtext {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result {
        font-size: 1.8rem;
        font-weight: bold;
        color: #22C55E;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #A1A1AA;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">🍌 Banana Ripeness Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload an image of a banana to see if it\'s <b>Underripe</b>, <b>Ripe</b>, or <b>Overripe</b>.</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("banana_ripeness_model.h5")

model = load_model()

# Class labels (based on your training)
class_names = ['overripe', 'ripe', 'underripe']

# Upload
uploaded_file = st.file_uploader("📤 Upload a banana image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Banana Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100
    predicted_class = class_names[predicted_index]

    # Show result
    st.markdown(f'<div class="result">🧠 Prediction: {predicted_class.upper()}<br>🎯 Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

# Footer
st.markdown(f'<div class="footer">Made with ❤️ by Avishkar Mulik • {datetime.now().year}</div>', unsafe_allow_html=True)

