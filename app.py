import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Define class names
class_names = ['Healthy', 'Unhealthy']

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title('Plant Disease Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    confidence = prediction[0][class_index]
    
    st.write(f"Classification: {class_name}")
    st.write(f"Confidence: {confidence:.2f}")

    st.subheader("Preprocessed Image")
    st.image(processed_image[0], caption='Preprocessed Image', use_column_width=True)