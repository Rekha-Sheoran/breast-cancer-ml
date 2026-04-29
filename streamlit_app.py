import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model/model.pkl', 'rb'))

st.title("🧬 Breast Cancer Prediction System")

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
