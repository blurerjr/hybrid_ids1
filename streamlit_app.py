import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model
import os # For handling file paths

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="NSL-KDD Intrusion Detection",
    layout="centered", # or "wide" depending on preference
    initial_sidebar_state="expanded" # Set to expanded by default to show inputs
)

# --- Model and Preprocessing Files Download ---
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
        response = requests.get(base_url + "scaler.pkl")
        if response.status_code == 200:
            with open(scaler_path, "wb") as f:
                f.write(response.content)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            st.success("Scaler loaded.")
        else:
            st.error(f"Failed to download scaler: {response.status_code}")
            st.stop()

    # Download and load selected features
    with st.spinner("Downloading and loading selected features..."):
        response = requests.get(base_url + "selected_features.pkl")
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
        response = requests.get(base_url + "encoder.h5")
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
        response = requests.get(base_url + "rf_model.pkl")
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

# Load all resources at the start of the app execution
scaler, selected_features, encoder, rf_model = load_resources()

# Label mapping for NSL-KDD (adjust based on your model's output encoding)
label_map = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

# --- Streamlit App UI ---
st.title("Network Intrusion Detection System")
st.markdown(
    """
    This application leverages a pre-trained machine learning model, enhanced by an Autoencoder,
    to predict network intrusion types based on connection features.
    """
)

# --- Sidebar: Input Features and Prediction Button ---
with st.sidebar:
    st.header("Enter Connection Details")
    st.markdown("Adjust the parameters below and click 'Predict' to classify the network traffic.")

    # Initialize a dictionary to hold all feature values.
    # All features expected by the model (from `selected_features`) are initialized to 0.
    input_features_dict = {feature: 0 for feature in selected_features}

    # Attack Type
    attack_options_display = ['Other', 'neptune', 'normal', 'satan']
    attack_selected_label = st.selectbox(
        "Attack Type:",
        attack_options_display,
        index=0,
        help="Select the type of attack, if known."
    )
    if attack_selected_label == 'neptune' and 'attack_neptune' in selected_features:
        input_features_dict['attack_neptune'] = 1
    elif attack_selected_label == 'normal' and 'attack_normal' in selected_features:
        input_features_dict['attack_normal'] = 1
    elif attack_selected_label == 'satan' and 'attack_satan' in selected_features:
        input_features_dict['attack_satan'] = 1

    # Numerical Inputs
    input_features_dict['count'] = st.number_input(
        "Connection Count (past 2 seconds):",
        value=0.0,
        format="%.2f",
        help="Number of connections to the same destination host in the last 2 seconds."
    )

    input_features_dict['dst_host_diff_srv_rate'] = st.number_input(
        "Dst Host Diff Srv Rate:",
        value=0.0,
        format="%.4f",
        help="Percentage of connections to different services at the destination host."
    )

    input_features_dict['dst_host_same_src_port_rate'] = st.number_input(
        "Dst Host Same Src Port Rate:",
        value=0.0,
        format="%.4f",
        help="Percentage of connections to the same source port at the destination host."
    )

    input_features_dict['dst_host_same_srv_rate'] = st.number_input(
        "Dst Host Same Srv Rate:",
        value=0.0,
        format="%.4f",
        help="Percentage of connections to the same service at the destination host."
    )

    input_features_dict['dst_host_srv_count'] = st.number_input(
        "Dst Host Service Count:",
        value=0.0,
        format="%.2f",
        help="Number of connections to the same service at the destination host."
    )

    # Flag Type
    flag_options_display = ['Other', 'S0', 'SF']
    flag_selected_label = st.selectbox(
        "Connection Status Flag:",
        flag_options_display,
        index=0,
        help="Status of the connection (Normal or Error)."
    )
    if flag_selected_label == 'S0' and 'flag_S0' in selected_features:
        input_features_dict['flag_S0'] = 1
    elif flag_selected_label == 'SF' and 'flag_SF' in selected_features:
        input_features_dict['flag_SF'] = 1

    input_features_dict['last_flag'] = st.number_input(
        "Last Flag:",
        value=0.0,
        format="%.2f",
        help="The value of the last flag observed in the connection."
    )

    # Logged In (binary)
    logged_in_options_display = ['0 (Not Logged In)', '1 (Logged In)']
    logged_in_selected_str = st.selectbox(
        "Logged In Status:",
        options=logged_in_options_display,
        index=0,
        help="1 if successfully logged in; 0 otherwise."
    )
    input_features_dict['logged_in'] = int(logged_in_selected_str.split(' ')[0])

    input_features_dict['same_srv_rate'] = st.number_input(
        "Same Service Rate:",
        value=0.0,
        format="%.4f",
        help="Percentage of connections that were to the same service."
    )

    input_features_dict['serror_rate'] = st.number_input(
        "SYN Error Rate:",
        value=0.0,
        format="%.4f",
        help="Percentage of connections with SYN errors."
    )

    # Service HTTP (binary)
    service_http_options_display = ['No', 'Yes']
    service_http_selected_label = st.selectbox(
        "HTTP Service Used:",
        options=service_http_options_display,
        index=0,
        help="Indicates if the destination network service used HTTP."
    )
    input_features_dict['service_http'] = 1 if service_http_selected_label == 'Yes' else 0

    st.markdown("---") # Separator in sidebar
    # The Predict button
    if st.button("Predict Attack Class in Sidebar"): # Changed button label to distinguish
        # This part of the code will execute when the button is clicked.
        # It's crucial to have the prediction logic here, but the result display
        # will appear in the main area.
        st.session_state['predict_clicked'] = True # Use session state to trigger display in main area
        st.session_state['input_features_dict'] = input_features_dict # Store inputs for prediction

# --- Main Area: Prediction Result & About Data ---
st.subheader("Prediction Result:")
# Check if prediction button was clicked
if 'predict_clicked' in st.session_state and st.session_state['predict_clicked']:
    input_features_dict_from_sidebar = st.session_state['input_features_dict']

    # Create a DataFrame from the collected feature values.
    input_df = pd.DataFrame([input_features_dict_from_sidebar])

    # Reorder columns to match the `selected_features` list
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

    st.success(f"The predicted attack class is: **{predicted_label}**")
    # Reset the flag after displaying result
    st.session_state['predict_clicked'] = False
else:
    st.info("Adjust parameters in the sidebar and click 'Predict' to see the result here.")


st.markdown("---") # Separator in main area

st.header("About the Dataset")
st.markdown(
    """
    This application utilizes the **NSL-KDD dataset**, a widely recognized benchmark for evaluating
    Intrusion Detection Systems. Below, you can see a preview of the first 10 rows of the dataset
    along with their respective column headings.
    """
)

# Load and display the top 10 rows of the dataset
data_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/KDDTrain%2B.txt"
columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
        "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate","attack","last_flag"]
try:
    data = pd.read_csv(data_url, names=columns, nrows=10) # Read only the first 10 rows
    st.dataframe(data)
except Exception as e:
    st.error(f"Error loading data: {e}")

st.markdown("[Link to full dataset](https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/KDDTrain%2B.txt)")

