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
    layout="wide", # Changed to wide for more space with sidebar and main content
    initial_sidebar_state="expanded"
)

# Define the 15 selected features for your model input
# This list is crucial for filtering and ordering the inputs
MODEL_INPUT_FEATURES = [
    'attack_buffer_overflow', 'attack_neptune', 'attack_normal',
    'attack_warezclient', 'count', 'dst_host_same_src_port_rate',
    'dst_host_same_srv_rate', 'dst_host_srv_count', 'flag_S0', 'flag_SF',
    'last_flag', 'logged_in', 'same_srv_rate', 'serror_rate',
    'service_http'
]

# --- Data Loading and Preprocessing for Slider Ranges ---
# This will load the full dataset to get min/max/median for sliders
@st.cache_data
def load_and_analyze_data():
    """
    Loads the full KDDTrain+ dataset to calculate statistics for input sliders.
    """
    data_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/KDDTrain%2B.txt"
    # Column names for the NSL-KDD dataset
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
               "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
               "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
               "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
               "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
               "dst_host_srv_rerror_rate", "attack", "last_flag"]
    
    try:
        df = pd.read_csv(data_url, names=columns)
        
        # --- Feature Engineering for one-hot encoded features (as they appear in your selected_features) ---
        # Ensure 'attack_buffer_overflow', 'attack_neptune', 'attack_normal', 'attack_warezclient'
        # 'flag_S0', 'flag_SF', 'service_http' are handled as binary (0/1) for stats
        
        # Attack features
        df['attack_buffer_overflow'] = (df['attack'] == 'buffer_overflow').astype(int)
        df['attack_neptune'] = (df['attack'] == 'neptune').astype(int)
        df['attack_normal'] = (df['attack'] == 'normal').astype(int)
        df['attack_warezclient'] = (df['attack'] == 'warezclient').astype(int)

        # Flag features
        df['flag_S0'] = (df['flag'] == 'S0').astype(int)
        df['flag_SF'] = (df['flag'] == 'SF').astype(int)
        
        # Service features
        df['service_http'] = (df['service'] == 'http').astype(int) # This should match how it was created for training

        # Select only the relevant features for min/max/median calculation
        # Filter to only include features that are actually in MODEL_INPUT_FEATURES
        # and are numerical for slider purposes.
        numerical_features_for_stats = [f for f in MODEL_INPUT_FEATURES if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        
        stats_df = df[numerical_features_for_stats].describe().loc[['min', 'max', '50%']].transpose()
        # Handle potential infinite or NaN values from describe()
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan)
        # Fill NaN for min/max/median with 0 or a sensible default if feature is constant
        for col in ['min', 'max', '50%']:
            stats_df[col] = stats_df[col].fillna(0) # or median if it's a constant value

        return df, stats_df
    except Exception as e:
        st.error(f"Error loading or analyzing data for slider ranges: {e}")
        st.stop() # Stop the app if data can't be loaded

data_load_state = st.info("Loading NSL-KDD dataset and calculating feature ranges...")
full_df, feature_stats = load_and_analyze_data()
data_load_state.empty() # Clear loading message

# --- Model and Preprocessing Files Download ---
@st.cache_resource
def load_resources():
    """
    Downloads and loads all necessary models and preprocessing files from GitHub.
    Uses st.cache_resource to cache these expensive operations.
    """
    base_url = "https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/"

    scaler_path = "scaler.pkl"
    encoder_path = "encoder.h5"
    rf_path = "rf_model.pkl"
    
    # selected_features is already defined as MODEL_INPUT_FEATURES
    # This might be redundant if selected_features.pkl is identical to MODEL_INPUT_FEATURES
    # If selected_features.pkl contains the *exact* list, then load it.
    # Otherwise, rely on MODEL_INPUT_FEATURES.
    # For now, we'll assume MODEL_INPUT_FEATURES is the source of truth for features.
    
    scaler_loaded, encoder_loaded, rf_model_loaded = None, None, None

    with st.spinner("Downloading and loading scaler..."):
        try:
            response = requests.get(base_url + "scaler.pkl")
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(scaler_path, "wb") as f:
                f.write(response.content)
            with open(scaler_path, "rb") as f:
                scaler_loaded = pickle.load(f)
            #st.success("Scaler loaded.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download scaler: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load scaler: {e}")
            st.stop()

    with st.spinner("Downloading and loading encoder model..."):
        try:
            response = requests.get(base_url + "encoder.h5")
            response.raise_for_status()
            with open(encoder_path, "wb") as f:
                f.write(response.content)
            encoder_loaded = load_model(encoder_path)
            #st.success("Encoder model loaded.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download encoder: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load encoder model: {e}")
            st.stop()

    with st.spinner("Downloading and loading Random Forest model..."):
        try:
            response = requests.get(base_url + "rf_model.pkl")
            response.raise_for_status()
            with open(rf_path, "wb") as f:
                f.write(response.content)
            with open(rf_path, "rb") as f:
                rf_model_loaded = pickle.load(f)
            #st.success("Random Forest model loaded.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download Random Forest model: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load Random Forest model: {e}")
            st.stop()
    
    return scaler_loaded, encoder_loaded, rf_model_loaded

# Load all resources at the start of the app execution
scaler, encoder, rf_model = load_resources()

# Label mapping for NSL-KDD (adjust based on your model's output encoding)
label_map = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

# --- Streamlit App UI ---
st.title("🛡️ Network Intrusion Detection System")
st.markdown(
    """
    This application utilizes a Hybrid approach for Network Intrusion Detection, combining an 
    Autoencoder for robust feature transformation and a powerful Random Forest Classifier 
    to accurately identify and categorize various network intrusion types.
    """
)

# --- Sidebar: Input Features and Prediction Button ---
with st.sidebar:
    st.header("⚙️ Configure Network Traffic Parameters")
    st.markdown("Adjust the features below to simulate network connection details and predict its status.")

    # Initialize a dictionary to hold all feature values, set to 0 by default.
    input_features_dict = {feature: 0 for feature in MODEL_INPUT_FEATURES}

    # --- Input fields for the 15 specified features ---

    # Categorical/Binary Features (use selectbox)
    st.subheader("Categorical Features")

    # Attack Type (Handles one-hot encoding for specified attack types)
    attack_options_display = ['Other', 'buffer_overflow', 'neptune', 'normal', 'warezclient']
    attack_selected_label = st.selectbox(
        "Attack Type:",
        attack_options_display,
        index=0, # Default to 'Other'
        help="Select the type of attack, if known (sets corresponding one-hot encoded feature)."
    )
    if attack_selected_label == 'buffer_overflow':
        input_features_dict['attack_buffer_overflow'] = 1
    elif attack_selected_label == 'neptune':
        input_features_dict['attack_neptune'] = 1
    elif attack_selected_label == 'normal':
        input_features_dict['attack_normal'] = 1
    elif attack_selected_label == 'warezclient':
        input_features_dict['attack_warezclient'] = 1

    # Flag Type (Handles one-hot encoding for specified flags)
    flag_options_display = ['Other', 'S0', 'SF']
    flag_selected_label = st.selectbox(
        "Connection Status Flag:",
        flag_options_display,
        index=0, # Default to 'Other'
        help="Status of the connection (e.g., S0 for connection rejected, SF for successful completion)."
    )
    if flag_selected_label == 'S0':
        input_features_dict['flag_S0'] = 1
    elif flag_selected_label == 'SF':
        input_features_dict['flag_SF'] = 1

    # Logged In (binary)
    logged_in_options_display = ['0 (Not Logged In)', '1 (Logged In)']
    logged_in_selected_str = st.selectbox(
        "Logged In Status:",
        options=logged_in_options_display,
        index=0,
        help="1 if successfully logged in; 0 otherwise."
    )
    input_features_dict['logged_in'] = int(logged_in_selected_str.split(' ')[0])

    # Service HTTP (binary)
    service_http_options_display = ['No', 'Yes']
    service_http_selected_label = st.selectbox(
        "HTTP Service Used:",
        options=service_http_options_display,
        index=0,
        help="Indicates if the destination network service used HTTP."
    )
    input_features_dict['service_http'] = 1 if service_http_selected_label == 'Yes' else 0

    st.subheader("Numerical Features")
    # Numerical Inputs (use sliders with dynamic ranges)
    for feature in MODEL_INPUT_FEATURES:
        if feature in feature_stats.index and feature not in ['attack_buffer_overflow', 'attack_neptune', 'attack_normal', 
                                                              'attack_warezclient', 'flag_S0', 'flag_SF', 'logged_in', 'service_http']:
            
            min_val = float(feature_stats.loc[feature, 'min'])
            max_val = float(feature_stats.loc[feature, 'max'])
            median_val = float(feature_stats.loc[feature, '50%'])

            # Determine appropriate step size for sliders
            # For rates, smaller step. For counts, integer or larger step.
            if 'rate' in feature or feature == 'serror_rate':
                step = 0.0001
                format_str = "%.4f"
            elif 'count' in feature:
                step = 1.0 if (max_val - min_val) > 10 else 0.1 # Adjust step for counts
                format_str = "%.1f"
                if min_val == 0 and max_val == 0: # Handle cases where a feature might be all zeros
                    max_val = 1.0 # Give it a small range
                    median_val = 0.0
            else: # last_flag or others
                step = 0.01
                format_str = "%.2f"

            # Adjust min/max if they are too close to avoid slider issues
            if min_val == max_val:
                max_val = min_val + 0.001 if min_val == 0 else min_val * 1.01 # ensure a small range
                
            input_features_dict[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}:", # Nicer display name
                min_value=min_val,
                max_value=max_val,
                value=median_val, # Default to median
                step=step,
                format=format_str,
                key=f"sidebar_slider_{feature}" # Unique key for each slider
            )

    st.markdown("---") # Separator in sidebar
    # The Predict button
    if st.button("🚀 Predict Network Status"):
        st.session_state['predict_clicked'] = True
        st.session_state['input_features_dict'] = input_features_dict
    # --- Main Area: Prediction Result & About Data ---
    st.subheader("📊 Prediction Result:")
    # Check if prediction button was clicked
    if 'predict_clicked' in st.session_state and st.session_state['predict_clicked']:
        input_features_dict_from_sidebar = st.session_state['input_features_dict']

        # Create a DataFrame from the collected feature values.
        input_df = pd.DataFrame([input_features_dict_from_sidebar])

        # Reorder columns to match the `MODEL_INPUT_FEATURES` list
        try:
            input_df = input_df[MODEL_INPUT_FEATURES]
        except KeyError as e:
            st.error(f"Error: A feature expected by the model is missing. Please check `MODEL_INPUT_FEATURES`. Missing key: {e}")
            st.stop()

        # Convert to numpy array for scaling
        input_data_for_scaling = input_df.values

        # Preprocess input: scale and then encode
        with st.spinner("Preprocessing input..."):
            input_data_scaled = scaler.transform(input_data_for_scaling)
            input_data_encoded = encoder.predict(input_data_scaled)
        #st.success("Input preprocessed.")

        # Make prediction
        with st.spinner("Making prediction..."):
            prediction = rf_model.predict(input_data_encoded)
            predicted_label = label_map.get(prediction[0], "Unknown")
    
        st.markdown("---") # Separator for result section

        # --- Beautify and Display Prediction (similar to guideline) ---
        if 'normal' in predicted_label.lower():
            st.success(f"### Detected Activity: **{predicted_label.upper()}** ✅")
            st.info("The model detects normal, non-intrusive network activity.")
        else:
            st.warning(f"### Detected Activity: **{predicted_label.upper()}** 🚨")
            st.error(f"**Potential Intrusion Detected!** Type: **{predicted_label}**. Immediate investigation recommended.")
            st.info("Proceed accordingly to further secure the system.")

        # Reset the flag after displaying result
        st.session_state['predict_clicked'] = False
    else:
        st.info("Adjust parameters in the sidebar and click 'Predict Network Status' to see the classification result here.")


st.markdown("---") # Separator in main area

st.header("📚 About the NSL-KDD Dataset")
st.markdown(
    """
    This application utilizes the **NSL-KDD dataset**, a widely recognized benchmark for evaluating
    Intrusion Detection Systems. It is a refined version of the KDD'99 dataset, addressing some
    of its inherent problems, thereby providing a more realistic evaluation of NIDS algorithms.
    """
)

# Load and display the top 10 rows of the dataset
st.subheader("First 10 Rows of NSL-KDD Training Data")
# Use the already loaded full_df for display
try:
    st.dataframe(full_df.head(10))
except Exception as e:
    st.error(f"Error displaying dataset preview: {e}")

st.markdown("[🔗 Link to full NSL-KDD Training Data (KDDTrain+.txt)](https://raw.githubusercontent.com/blurerjr/hybrid_ids1/refs/heads/master/KDDTrain%2B.txt)")

st.markdown("---")
st.markdown("Developed with ❤️ for Network Intrusion Detection Systems")
