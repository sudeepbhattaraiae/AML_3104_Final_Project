import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import tensorflow as tf

# Set a background image URL
background_image = 'https://raw.githubusercontent.com/Bijay555/AML_3104_Final_Project/main/images/leaf_image1.jpg'

# Define HTML code with CSS to apply the background image
html_code = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url('{background_image}');
    background-size: 110% !important;
    background-position: top left !important;
    background-repeat: repeat !important;
    background-attachment: local !important;
}}
</style>
"""

# Display the HTML code in Streamlit
st.markdown(html_code, unsafe_allow_html=True)

st.title("Plant Leaf Disease Detection")
st.header("About the Project")
st.write("This application is designed to assist in the detection of plant leaf diseases using advance deep learning models. User can upload an image of a plant leaf and the app will classify the disease present in the leaf, providing a confidence score for the prediction.")

st.write("The app offers two models for disease detection: ")
st.markdown("""
- CNN Model
- Xception Model
""")

st.write("This flexibility ensures accurate results tailored to specific needs and datasets.")
# Load the model
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
xception_model = tf.keras.models.load_model('models/cnn_model.h5')



# Define class names (you might need to adjust this based on your actual class names)
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Common_rust',
 'Gray_leaf_spot',
 'Northern_Leaf_Blight',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy',
 'healthy']  

def preprocess_image(image):
    # Ensure the image is in RGB format (3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = image.resize((224, 224))  # Adjust size as per your model's input requirements
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array

def predict(model, img):
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence



# sidebar for navigation
with st.sidebar:
    selected = option_menu('Prediction Models:',
                          
                          ['Select a model',
                           'Xception Model',
                           'CNN model'],
                          default_index=0)
    

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Select model based on user choice
    if selected == 'Xception Model':
        model = xception_model
    elif selected == 'CNN model':
        model = cnn_model
    else:
        st.warning("Please select a model from the sidebar.")
        st.stop()

    if st.button('Predict'):
        try:
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make prediction
            predicted_class, confidence = predict(model, processed_image)
            
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence}%")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.warning("Please select the model to get the prediction.")

else:
    st.warning("Please upload an image first.")

