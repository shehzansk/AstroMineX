import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("./RF_mining_model.pkl")

# Title of the web app
st.title("Mining Site Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")

# Function to get user input from sidebar sliders
def user_input_features():
    Distance_from_Earth = st.sidebar.slider("Distance from Earth (M km)", 1.0, 1000.0, 100.0)
    Iron = st.sidebar.slider("Iron (%)", 0.0, 100.0, 50.0)
    Nickel = st.sidebar.slider("Nickel (%)", 0.0, 100.0, 50.0)
    Water_Ice = st.sidebar.slider("Water Ice (%)", 0.0, 100.0, 50.0)
    Other_Minerals = st.sidebar.slider("Other Minerals (%)", 0.0, 100.0, 50.0)
    Estimated_Value = st.sidebar.slider("Estimated Value (B USD)", 0.0, 500.0, 100.0)
    Sustainability_Index = st.sidebar.slider("Sustainability Index", 0.0, 100.0, 0.5)
    Efficiency_Index = st.sidebar.slider("Efficiency Index", 0.0, 100.0, 0.5)

    # Store user inputs in a dictionary
    data = {
        'Distance from Earth (M km)': Distance_from_Earth,
        'Iron (%)': Iron,
        'Nickel (%)': Nickel,
        'Water Ice (%)': Water_Ice,
        'Other Minerals (%)': Other_Minerals,
        'Estimated Value (B USD)': Estimated_Value,
        'Sustainability Index': Sustainability_Index,
        'Efficiency Index': Efficiency_Index
    }

    # Create a DataFrame using user inputs
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

# Display the prediction result
st.subheader('Prediction Result')

# Customize the prediction message
if prediction[0] == 1:
    st.success("✅ This is a Potential Mining Site.")
else:
    st.error("❌ This is Not a Potential Mining Site.")

# Optionally, you can add more details or a description below the result
st.markdown("""
Note: The prediction is based on the model's analysis of key features such as distance from Earth, mineral composition, estimated value (B USD), and sustainability indices.
""", unsafe_allow_html=True)
