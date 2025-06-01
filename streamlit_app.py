import requests
import pickle
import streamlit as st

# Download the preprocessing file from GitHub
scaler_url = "https://raw.githubusercontent.com/username/repo/main/scaler.pkl"
response = requests.get(scaler_url)
with open("scaler.pkl", "wb") as f:
    f.write(response.content)

# Load the preprocessing file
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example: Download the model file (as you provided)
model_url = "https://raw.githubusercontent.com/username/repo/main/model.pkl"
response = requests.get(model_url)
with open("model.pkl", "wb") as f:
    f.write(response.content)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.title("Prediction App")
input_data = st.number_input("Enter input feature", value=0.0)

# Preprocess input data
input_data_scaled = scaler.transform([[input_data]])  # Adjust based on your preprocessing

# Make prediction
prediction = model.predict(input_data_scaled)
st.write(f"Prediction: {prediction[0]}")
