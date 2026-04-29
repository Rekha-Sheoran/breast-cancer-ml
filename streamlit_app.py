import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model/model.pkl', 'rb'))

st.title("🧬 Breast Cancer Prediction System")

st.subheader("Enter Tumor Details")
mean_radius = st.slider("Mean Radius", 5.0, 30.0, 14.0)
mean_texture = st.slider("Mean Texture", 10.0, 40.0, 20.0)
mean_perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 90.0)
mean_area = st.slider("Mean Area", 200.0, 2500.0, 600.0)
mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.2, 0.1)

st.write("Enter all 30 feature values separated by commas")


input_data = st.text_area("Input Features")

if st.button("Predict"):
    try:
        data = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)

        if data.shape[1] != 30:
            st.error("❌ Please enter exactly 30 values")
        else:
            prediction = model.predict(data)[0]

            if prediction == 0:
                st.error("❌ Malignant (Cancerous)")
            else:
                st.success("✅ Benign (Non-cancerous)")

    except:
        st.warning("⚠️ Invalid input. Please enter only numbers separated by commas.")
