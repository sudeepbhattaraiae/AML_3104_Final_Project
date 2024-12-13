import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")

# Set a background image URL
background_image = 'https://raw.githubusercontent.com/kandelsatish/Walmart_Sales_Prediction/main/static/images/drk.png'
html_code  = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url('{background_image}');
background-size: 110%;
background-position: top left;
background-repeat:repeat;
background-attachment: local;
}}
</style>
"""

# Display the HTML code using st.markdown
st.markdown(html_code, unsafe_allow_html=True)


# Load the model
cnn_model = tf.keras.models.load_model('plant_disease_model.h5')
random_forest_model = tf.keras.models.load_model('plant_disease_model.h5')
Logistic_regression_model = tf.keras.models.load_model('plant_disease_model.h5')
svm_model = tf.keras.models.load_model('plant_disease_model.h5')


# Define class names (you might need to adjust this based on your actual class names)
class_names = ['Class1', 'Class2', 'Class3', ...]  # Replace with your actual class names

def preprocess_image(image):
    img = image.resize((224, 224))  # Adjust size as per your model's input requirements
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array

def predict(model, img):
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

st.title("Plant Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predicted_class, confidence = predict(model, processed_image)
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence}%")

    st.subheader("Preprocessed Image")
    st.image(processed_image[0], caption='Preprocessed Image', use_column_width=True)

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Prediction Models:',
                          
                          ['Select a model',
                           'Random Forest Regressor',
                           'ElastiNet Model',
                           'Ridge Regression Model',
                           'Lasso Regression Model',
                           'Linear Regression Model'],
                          default_index=0)
    
# Make prediction using the model:
st.header('Prediction of Weekly sales of Walmart: ')
if (selected == 'Random Forest Regressor'):
    model = random_forest_reg_model
     
elif(selected == 'Logistic Regression Model'):
    model = Ridge_regression_model
    
elif(selected == 'CNN model'):
    model = Lasso_regression_model
    
elif(selected == 'SVM model'):
    model = Linear_regression_model
  