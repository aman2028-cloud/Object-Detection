import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ======================
# Load TFLite model
# ======================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_float16_32.tflite", experimental_delegates=[])
    interpreter.allocate_tensors()
    return interpreter

# ======================
# Preprocess image
# ======================
def preprocess_batch_images(image: Image.Image) -> np.ndarray:
    image = image.resize((32, 32))
    img_array = np.array(image).astype(np.float32) / 255.0  # (32, 32, 3)
    
    # Repeat the image 32 times → shape: (32, 32, 32, 3)
    batch_array = np.stack([img_array] * 32, axis=0)  # shape: (32, 32, 32, 3)
    return batch_array



# ======================
# Make prediction
# ======================
def predict(img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_batch_images(img).astype(input_details[0]['dtype'])
    interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output_data)
    confidence = np.max(output_data)
    return prediction, confidence

# ======================
# CIFAR-10 Labels
# ======================
cifar10_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# ======================
# Streamlit UI
# ======================
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image  for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

interpreter = load_tflite_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying.."):
        pred_class, conf = predict(image)
        st.success(f"**Prediction:** {cifar10_labels[pred_class]} ({conf*100:.2f}% confidence)")

