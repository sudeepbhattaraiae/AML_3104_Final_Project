import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Set a background image URL
background_image = 'images/leaf_image1.jpg'
html_code  = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url('{background_image}');
background-size: 110%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
</style>
"""

# Display the HTML code using st.markdown
st.markdown(html_code, unsafe_allow_html=True)

# Load the models
cnn_model = tf.keras.models.load_model('./models/Model_P_v8.h5')
# random_forest_model = tf.keras.models.load_model('random_forest_model.h5')
# logistic_regression_model = tf.keras.models.load_model('logistic_regression_model.h5')
# svm_model = tf.keras.models.load_model('svm_model.h5')

# Define class names (update based on your actual class labels)
class_names = ['Healthy', 'Diseased']  # Replace with your specific classes

def preprocess_image(image):
    try:
        img = image.resize((224, 224))  # Adjust size as per your model's input requirements
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        return img_array
    except Exception as e:
        st.error(f"Error in preprocessing the image: {e}")
        return None

def predict(model, img):
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

st.title("Plant Leaf Disease Detection")

# Description of the app
st.markdown("""
### About the Project
This application is designed to assist in the detection of plant leaf diseases using advanced machine learning models. Users can upload an image of a plant leaf, and the app will classify it as healthy or diseased, providing a confidence score for the prediction. 

The app allows you to choose from multiple prediction models, including:
- Convolutional Neural Network (CNN)
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

This flexibility ensures accurate results tailored to specific needs and datasets.
""")

# Sidebar for model selection
st.sidebar.title("Select Prediction Model")
model_option = st.sidebar.selectbox("Choose a model", ["CNN", "Random Forest", "Logistic Regression", "SVM"])

# Select model based on user choice
if model_option == "CNN":
    model = cnn_model
elif model_option == "Random Forest":
    model = random_forest_model
elif model_option == "Logistic Regression":
    model = logistic_regression_model
elif model_option == "SVM":
    model = svm_model

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Check image size
    if image.size[0] > 4096 or image.size[1] > 4096:  # Example threshold
        st.error("The uploaded image is too large. Please upload an image smaller than 4096x4096 pixels.")
    else:
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        if processed_image is not None:
            # Make prediction
            predicted_class, confidence = predict(model, processed_image)

            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence}%")

            st.subheader("Preprocessed Image")
            # Convert TensorFlow tensor to NumPy array for display
            preprocessed_np_image = processed_image[0].numpy() / 255.0
            st.image(preprocessed_np_image, caption='Preprocessed Image', use_container_width=True)
