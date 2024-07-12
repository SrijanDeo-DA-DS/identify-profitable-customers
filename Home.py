import streamlit as st


st.set_page_config(
    page_title="Streamlit Dark Mode Example",
    page_icon=":dark_mode:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "This is a Streamlit app with a dark mode theme!"
    }
)

# Add your Streamlit app content here
st.title('Customer Profiling Dashboard')



st.subheader('''
         Welcome to Customer Profiling Dashboard. You can select from 4 options :
         1. Analytics - Data Analysis on the customers
         2. Model Explainability - Understand what factors impact for a Customer to be Profitable vs Non-Profitable
         3. Predictor - Predict whether a given customer is Profitable or not
         4. Recommendation - On the basis of selected customer, the model will output top 5 similar customers
         ''')