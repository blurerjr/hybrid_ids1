import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model

# Download models and preprocessing files from GitHub
scaler_url = "https://raw.githubusercontent.com/username/repo/main/scaler.pkl"
features_url = "https://raw.githubusercontent.com/username/repo/main/selected_features.pkl"
encoder_url = "https://raw.githubusercontent.com/username/repo/main/encoder.h5"
rf_url = "https://raw.githubusercontent.com/username/repo/main/rf_model.pkl"

# Load scaler
response = requests.get(scaler_url)
with open("scaler.pkl", "wb") as f:
    f.write(response.content)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load selected features
response = requests.get(features_url)
with open("selected_features.pkl", "wb") as f:
    f.write(response.content)
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Load encoder
response = requests.get(encoder_url)
with open("encoder.h5", "wb") as f:
    f.write(response.content)
encoder = load_model("encoder.h5")

# Load Random Forest
response = requests.get(rf_url)
with open("rf_model.pkl", "wb") as f:
    f.write(response.content)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Label mapping for NSL-KDD (adjust based on your attack_class encoding)
label_map = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

# Streamlit app
st.title("NSL-KDD Intrusion Detection")

# Collect input for the 15 features
input_data = []
for feature in selected_features:
    if feature in ['attack_neptune', 'attack_normal', 'attack_satan', 'flag_S0', 
                   'flag_SF', 'service_http', 'logged_in']:
        value = st.selectbox(f"Select {feature}", [0, 1], index=0)
    else:
        value = st.number_input(f"Enter {feature}", value=0.0)
    input_data.append(value)

# Preprocess input
input_data = np.array([input_data])
input_data_scaled = scaler.transform(input_data)
input_data_encoded = encoder.predict(input_data_scaled)

# Predict
if st.button("Predict"):
    prediction = rf_model.predict(input_data_encoded)
    predicted_label = label_map.get(prediction[0], "Unknown")
    st.write(f"Prediction: {predicted_label}")
