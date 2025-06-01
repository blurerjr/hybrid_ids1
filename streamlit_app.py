import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model
import os # For handling file paths

# --- Streamlit Page Configuration ---
# This MUST be the very first Streamlit command in your script.
st.set_page_config(
    page_title="NSL-KDD Intrusion Detection",
    layout="centered", # or "wide" depending on preference
    initial_sidebar_state="auto" # or "expanded", "collapsed"
)

# --- Model and Preprocessing Files Download ---
# Using st.cache_resource to avoid re-downloading and re-loading on every rerun
# This significantly improves performance on Streamlit Cloud.
@st.cache_resource
def load_resources():
    """
    Downloads and loads all necessary models and preprocessing files from GitHub.
    Uses st.cache_resource to cache these expensive operations.
    """
    base_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/"
    
    # Define local paths for downloaded files
    scaler_path = "scaler.pkl"
    features_path = "selected_features.pkl"
    encoder_path = "encoder.h5"
    rf_path = "rf_model.pkl"

    # Download and load scaler
    with st.spinner("Downloading and loading scaler..."):
        response = requests.get(scaler_url)
        if response.status_code == 200:
            with open(scaler_path, "wb") as f:
                f.write(response.content)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            st.success("Scaler loaded.")
        else:
            st.error(f"Failed to download scaler: {response.status_code}")
            st.stop() # Stop the app if a critical resource fails to load

    # Download and load selected features
    with st.spinner("Downloading and loading selected features..."):
        response = requests.get(features_url)
        if response.status_code == 200:
            with open(features_path, "wb") as f:
                f.write(response.content)
            with open(features_path, "rb") as f:
                selected_features = pickle.load(f)
            st.success("Selected features loaded.")
        else:
            st.error(f"Failed to download selected features: {response.status_code}")
            st.stop()

    # Download and load encoder
    with st.spinner("Downloading and loading encoder model..."):
        response = requests.get(encoder_url)
        if response.status_code == 200:
            with open(encoder_path, "wb") as f:
                f.write(response.content)
            encoder = load_model(encoder_path)
            st.success("Encoder model loaded.")
        else:
            st.error(f"Failed to download encoder model: {response.status_code}")
            st.stop()

    # Download and load Random Forest
    with st.spinner("Downloading and loading Random Forest model..."):
        response = requests.get(rf_url)
        if response.status_code == 200:
            with open(rf_path, "wb") as f:
                f.write(response.content)
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)
            st.success("Random Forest model loaded.")
        else:
            st.error(f"Failed to download Random Forest model: {response.status_code}")
            st.stop()
    
    return scaler, selected_features, encoder, rf_model

# URLs for models and preprocessing files (defined here for clarity, used in load_resources)
scaler_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/scaler.pkl"
features_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/selected_features.pkl"
encoder_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/encoder.h5"
rf_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/rf_model.pkl"

# Load all resources at the start of the app execution
scaler, selected_features, encoder, rf_model = load_resources()

# Label mapping for NSL-KDD (adjust based on your model's output encoding)
# This mapping assumes the Random Forest model outputs integer labels 0-4
# corresponding to these attack types.
label_map = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

# --- Streamlit App UI ---
st.title("Network Intrusion Detection System")
st.markdown(
    """
    This application predicts the type of network intrusion based on connection
    features using a pre-trained machine learning model.
    """
)

st.header("Enter Connection Details")

# Initialize a dictionary to hold all feature values.
# All features expected by the model (from `selected_features`) are initialized to 0.
# This is crucial for one-hot encoded features that might not be explicitly selected.
input_features_dict = {feature: 0 for feature in selected_features}

# --- Input fields matching the HTML form structure ---

# Attack Type (categorical, will be one-hot encoded)
attack_options_display = ['Other', 'neptune', 'normal', 'satan']
attack_selected_label = st.selectbox(
    "Attack (from initial HTML form's 'attack' field):",
    attack_options_display,
    index=0 # Default to 'Other'
)
# Map selected display label to the corresponding one-hot encoded feature name
if attack_selected_label == 'neptune' and 'attack_neptune' in selected_features:
    input_features_dict['attack_neptune'] = 1
elif attack_selected_label == 'normal' and 'attack_normal' in selected_features:
    input_features_dict['attack_normal'] = 1
elif attack_selected_label == 'satan' and 'attack_satan' in selected_features:
    input_features_dict['attack_satan'] = 1
# If 'Other' is selected, or if the feature is not in `selected_features`,
# its value remains 0 as initialized.

# Numerical Inputs
input_features_dict['count'] = st.number_input(
    "Number of connections to the same destination host as the current connection in the past two seconds:",
    value=0.0,
    format="%.2f",
    help="e.g., 10.0"
)

input_features_dict['dst_host_diff_srv_rate'] = st.number_input(
    "The percentage of connections that were to different services, among the connections aggregated in dst_host_count :",
    value=0.0,
    format="%.4f",
    help="e.g., 0.15"
)

input_features_dict['dst_host_same_src_port_rate'] = st.number_input(
    "The percentage of connections that were to the same source port, among the connections aggregated in dst_host_srv_count :",
    value=0.0,
    format="%.4f",
    help="e.g., 0.05"
)

input_features_dict['dst_host_same_srv_rate'] = st.number_input(
    "The percentage of connections that were to the same service, among the connections aggregated in dst_host_count :",
    value=0.0,
    format="%.4f",
    help="e.g., 0.80"
)

input_features_dict['dst_host_srv_count'] = st.number_input(
    "Number of connections having the same port number :",
    value=0.0,
    format="%.2f",
    help="e.g., 50.0"
)

# Flag Type (categorical, will be one-hot encoded)
flag_options_display = ['Other', 'S0', 'SF']
flag_selected_label = st.selectbox(
    "Status of the connection – Normal or Error (from initial HTML form's 'flag' field):",
    flag_options_display,
    index=0 # Default to 'Other'
)
# Map selected display label to the corresponding one-hot encoded feature name
if flag_selected_label == 'S0' and 'flag_S0' in selected_features:
    input_features_dict['flag_S0'] = 1
elif flag_selected_label == 'SF' and 'flag_SF' in selected_features:
    input_features_dict['flag_SF'] = 1
# If 'Other' is selected, or if the feature is not in `selected_features`,
# its value remains 0 as initialized.

input_features_dict['last_flag'] = st.number_input(
    "Last Flag :",
    value=0.0,
    format="%.2f",
    help="e.g., 0.0 or 1.0"
)

# Logged In (binary)
logged_in_options_display = ['0 (Not Logged In)', '1 (Logged In)']
logged_in_selected_str = st.selectbox(
    "1 if successfully logged in; 0 otherwise :",
    options=logged_in_options_display,
    index=0
)
# Extract the numerical value (0 or 1)
input_features_dict['logged_in'] = int(logged_in_selected_str.split(' ')[0])


input_features_dict['same_srv_rate'] = st.number_input(
    "The percentage of connections that were to the same service, among the connections aggregated in count :",
    value=0.0,
    format="%.4f",
    help="e.g., 0.90"
)

input_features_dict['serror_rate'] = st.number_input(
    "The percentage of connections that have activated the flag (4) s0, s1, s2 or s3, among the connections aggregated in count :",
    value=0.0,
    format="%.4f",
    help="e.g., 0.10"
)

# Service HTTP (binary)
service_http_options_display = ['No', 'Yes']
service_http_selected_label = st.selectbox(
    "Destination network service used http or not :",
    options=service_http_options_display,
    index=0
)
input_features_dict['service_http'] = 1 if service_http_selected_label == 'Yes' else 0


# --- Prediction Logic ---
if st.button("Predict Attack Class"):
    # Create a DataFrame from the collected feature values.
    # It's crucial to ensure the columns are in the exact order expected by the scaler and encoder.
    # The `selected_features` list provides this order.
    input_df = pd.DataFrame([input_features_dict])
    
    # Reorder columns to match the `selected_features` list
    # This step is critical if the order in `input_features_dict` doesn't strictly match.
    try:
        input_df = input_df[selected_features] 
    except KeyError as e:
        st.error(f"Error: A feature expected by the model is missing. Please check `selected_features.pkl`. Missing key: {e}")
        st.stop()


    # Convert to numpy array for scaling
    input_data_for_scaling = input_df.values

    # Preprocess input: scale and then encode
    with st.spinner("Preprocessing input..."):
        input_data_scaled = scaler.transform(input_data_for_scaling)
        input_data_encoded = encoder.predict(input_data_scaled)
    st.success("Input preprocessed.")

    # Make prediction
    with st.spinner("Making prediction..."):
        prediction = rf_model.predict(input_data_encoded)
        predicted_label = label_map.get(prediction[0], "Unknown")
    
    st.subheader("Prediction Result:")
    st.success(f"The predicted attack class is: **{predicted_label}**")

st.markdown("---")
st.markdown("Developed for Network Intrusion Detection System (NIDS) Web-App")

