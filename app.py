import streamlit as st

# Title of the app
st.title("Demo Streamlit App")

# Text input for user
user_input = st.text_input("Enter some text:")

# Button to submit the input
if st.button("Submit"):
    st.write("You entered:", user_input)