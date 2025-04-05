import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="banana_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image: Image.Image):
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = np.max(output_data)
    
    return prediction, confidence

labels = ['Overripe', 'Ripe', 'Underripe']

# Streamlit UI
st.set_page_config(page_title="Banana Ripeness Classifier", layout="centered")
st.title("🍌 Banana Ripeness Predictor")
st.write("Upload a banana image to check if it is underripe, ripe, or overripe.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Predicting..."):
        interpreter = load_tflite_model()
        preprocessed = preprocess_image(image)
        pred, conf = predict(preprocessed, interpreter)
        st.success(f"Prediction: **{labels[pred]}** with {conf*100:.2f}% confidence.")


# Footer
st.markdown(f'<div class="footer">Made with ❤️ by Avishkar Mulik • {datetime.now().year}</div>', unsafe_allow_html=True)

