"""
streamlit_dashboard/pages.py
Halaman prediksi untuk Brain MRI Tumor Classification
"""

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def page_prediction_upload(nn, conv, class_names):
    """Upload Image Prediction Tab"""
    st.subheader("Unggah Image Brain MRI")
    
    uploaded_file = st.file_uploader("Pilih file image (JPG, PNG):", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", width='stretch')
        
        with col2:
            # Preprocess image
            img_array = np.array(image.convert('L').resize((128, 128)))
            img_normalized = img_array / 255.0
            img_flat = img_array.flatten()
            
            # Extract features menggunakan Conv
            img_features = conv.extract_features(img_array)
            img_features = img_features.reshape(1, -1)  # Reshape untuk NN input
            
            # Prediction
            pred_proba = nn.predict_proba(img_features)
            pred_class = int(nn.predict(img_features).item())
            
            st.subheader("Prediction Results")
            
            if class_names:
                predicted_class = class_names[pred_class]
                confidence = pred_proba[0][pred_class] * 100
                
                st.success(f"**Predicted Class:** {predicted_class.upper()}")
                st.success(f"**Confidence:** {confidence:.2f}%")
                
                # Probability distribution
                st.subheader("Probability Distribution")
                
                # Tampilkan raw probability dengan lebih detail
                st.info(f"📊 **Raw Probabilities (sebelum × 100):**")
                prob_raw = {class_names[i]: f"{pred_proba[0][i]:.6f}" for i in range(len(class_names))}
                for cls, prob in prob_raw.items():
                    st.write(f"  {cls}: {prob}")
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                prob_prct = pred_proba[0] * 100
                colors = ['#27ae60' if c == predicted_class else '#95a5a6' for c in class_names]
                ax.barh(class_names, prob_prct, color=colors)
                ax.set_xlabel('Probability (%)', fontsize=12)
                ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
                ax.set_xlim([0, 100])
                
                for i, v in enumerate(prob_prct):
                    if v < 0.01:
                        label = f'{v:.2e}%'
                    else:
                        label = f'{v:.2f}%'
                    ax.text(v + 1 if v > 1 else 3, i, label, va='center', fontsize=9)
                
                st.pyplot(fig)


def page_prediction_dataset(nn, conv, class_names, dataset):
    """Test Dataset Prediction Tab"""
    st.subheader("Prediksi pada Test Dataset")
    
    if dataset:
        # Select class
        selected_class = st.selectbox("Pilih Class:", class_names)
        
        # Select sample
        class_idx = class_names.index(selected_class)
        class_data = dataset[selected_class]
        
        sample_idx = st.slider(f"Sample dari {selected_class}:", 0, len(class_data)-1, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display image
            img_sample = class_data[sample_idx]
            st.image(img_sample, caption=f"Sample: {selected_class}", width='stretch')
        
        with col2:
            # Prediction
            img_features = conv.extract_features(img_sample)
            img_features = img_features.reshape(1, -1)  # Reshape untuk NN input
            pred_proba = nn.predict_proba(img_features)
            pred_class = int(nn.predict(img_features).item())
            
            predicted_class_name = class_names[pred_class]
            confidence = pred_proba[0][pred_class] * 100
            ground_truth = selected_class
            is_correct = predicted_class_name == ground_truth
            
            st.subheader("Prediction Result")
            
            if is_correct:
                st.success(f"✅ **CORRECT!**")
            else:
                st.error(f"❌ **INCORRECT**")
            
            st.write(f"**Ground Truth:** {ground_truth}")
            st.write(f"**Predicted:** {predicted_class_name}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Probability distribution
            st.subheader("Probabilities")
            prob_data = {
                'Class': class_names,
                'Probability (Raw)': [f"{p:.8f}" for p in pred_proba[0]],
                'Probability (%)': [f"{p*100:.4f}%" for p in pred_proba[0]]
            }
            prob_df = pd.DataFrame(prob_data)
            st.dataframe(prob_df, width='stretch', hide_index=True)
