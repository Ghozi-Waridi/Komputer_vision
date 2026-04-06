"""
🧠 Brain MRI Tumor Classification - Prediction Interface
Simple UI untuk Prediksi Model Neural Network pada Brain MRI
"""

import streamlit as st
import numpy as np
import os
import glob
from datetime import datetime

from layers.Conv import Conv
from layers.NN import NeuralNetwork
from utils.model_saver import ModelSaver
from utils.load import load_all_images
from streamlit_dashboard import (
    page_prediction_upload,
    page_prediction_dataset
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-text {
        color: #27ae60;
        font-weight: bold;
    }
    .warning-text {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL SELECTION (Auto-load latest model)
# ============================================================================

saver = ModelSaver(model_dir='models')
model_files = sorted(glob.glob('models/*_nn_*.pkl'))

if not model_files:
    st.error("❌ Tidak ada model ditemukan di folder 'models/'")
    st.stop()

# Auto-load latest model (last file in sorted list)
model_path = model_files[-1]

# ============================================================================
# LOAD MODEL DATA
# ============================================================================

@st.cache_resource
def load_model(_model_path):
    """Load model dari pickle file"""
    model_data = saver.load_neural_network(_model_path)
    return model_data

@st.cache_resource
def load_results(_model_path):
    """Load training results"""
    results_path = _model_path.replace('_nn_', '_results_')
    if os.path.exists(results_path):
        return saver.load_training_results(results_path)
    return None

@st.cache_data
def load_dataset():
    """Load dataset untuk testing & demo"""
    IMAGE_SIZE = (128, 128)
    DATA_ROOT = 'data/raw'
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    try:
        images = load_all_images(DATA_ROOT, CLASS_NAMES, IMAGE_SIZE)
        return images, CLASS_NAMES
    except:
        return None, None


model_data = load_model(model_path)
results = load_results(model_path)

layer_sizes = model_data['layer_sizes']
nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=model_data['learning_rate'])
nn.weights = model_data['weights']
nn.biases = model_data['biases']

conv = Conv(pool_size=2)
dataset, CLASS_NAMES = load_dataset()

# ============================================================================
# MAIN PAGE - PREDICTION ONLY
# ============================================================================

st.markdown("# Brain MRI Tumor Classification - Prediction")

# Prediction tabs
pred_tab1, pred_tab2 = st.tabs([" Upload Image", "Test Dataset"])

with pred_tab1:
    page_prediction_upload(nn, conv, CLASS_NAMES)

with pred_tab2:
    if dataset:
        page_prediction_dataset(nn, conv, CLASS_NAMES, dataset)
    else:
        st.warning("Dataset not loaded. Please ensure data/raw/ folder exists with images.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 2rem;'>
    <p>🧠 Brain MRI Tumor Classification - Prediction Only</p>
    <p>Powered by Streamlit | Neural Network + Convolution Layer</p>
    <p><small>Model saved: {}</small></p>
</div>
""".format(model_data.get('timestamp', 'N/A')), unsafe_allow_html=True)
