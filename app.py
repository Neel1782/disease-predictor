import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import json
import pandas as pd
import plotly.graph_objects as go
from streamlit_tags import st_tags
from streamlit_lottie import st_lottie
import requests

# === Load Lottie animations ===
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_medical = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_Cc8Bpg.json")
lottie_loader = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_j1adxtyb.json")

# === Load assets ===
model = tf.keras.models.load_model("model/ann_model.h5")
with open("encoder/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("symptoms.txt", "r") as f:
    symptoms = [line.strip() for line in f]
with open("disease_info.json", "r") as f:
    disease_data = json.load(f)
with open("followup_questions.json", "r") as f:
    followup_data = json.load(f)

st.set_page_config(page_title="AI Disease Predictor", layout="wide", page_icon="🧬")

if "history" not in st.session_state:
    st.session_state["history"] = []

# === Custom Button Style ===
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0088cc;
    color:white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
    box-shadow: 0 0 10px #00c6ff;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 0 20px #00ffcc;
}
</style>
""", unsafe_allow_html=True)

# === Header ===
col1, col2 = st.columns([1, 3])
with col1:
    st_lottie(lottie_medical, height=140, key="header_anim")
with col2:
    st.markdown("<h1 style='margin-top:25px;'>🧬 Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px;'>Enter your symptoms and let AI assist with diagnosis</p>", unsafe_allow_html=True)

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["🩺 Predict", "📘 About", "📊 Model Info"])

# === Tab 1: Prediction ===
with tab1:
    input_method = st.radio("Input Method:", ["Multiselect", "Tag-style"], horizontal=True)

    if input_method == "Multiselect":
        selected_symptoms = st.multiselect("Select symptoms:", symptoms)
    else:
        selected_symptoms = st_tags(
            label="Enter symptoms (press enter):",
            text="",
            suggestions=symptoms,
            key="symptom_tags"
        )

    if st.button("🚑 Predict Disease", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom before predicting.")
        else:
            with st.spinner("🤖 AI is thinking..."):
                st_lottie(lottie_loader, height=120, key="thinking")
                input_vector = np.zeros(len(symptoms))
                for s in selected_symptoms:
                    if s in symptoms:
                        input_vector[symptoms.index(s)] = 1

                prediction = model.predict(np.array([input_vector]))
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_disease = le.inverse_transform([predicted_index])[0]
                confidence = np.max(prediction) * 100

                st.session_state["history"].append({
                    "symptoms": selected_symptoms,
                    "disease": predicted_disease,
                    "confidence": round(confidence, 2)
                })

            st.success(f"🎯 Predicted Disease: **{predicted_disease}**")

            # Confidence Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Model Confidence"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Disease Info Card
            if predicted_disease in disease_data:
                info = disease_data[predicted_disease]
                st.markdown(f"""
                <div style="
                    background: linear-gradient(to right, #ffffff, #f8f9fa);
                    border-radius: 12px;
                    padding: 20px 25px;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
                    color: #1c1c1c;
                    margin-top: 20px;
                    font-family: 'Segoe UI', sans-serif;
                ">
                    <h3 style="color:#0072ff; display: flex; align-items: center; gap: 10px;">
                        🧠 <span>Disease Info</span>
                    </h3>
                    <p style="margin-top: 10px; line-height: 1.6;">
                        <strong>Description:</strong> {info['description']}<br><br>
                        <strong>Treatment:</strong> {info['treatment']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Follow-up Questions
            if predicted_disease in followup_data:
                with st.expander("💬 Follow-up Questions"):
                    for q in followup_data[predicted_disease]:
                        st.radio(q, ["Yes", "No"], key=q)

    # Prediction History
    if st.session_state["history"]:
        with st.expander("🕓 Prediction History"):
            df = pd.DataFrame(st.session_state["history"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download History", csv, "prediction_history.csv", "text/csv")

# === Tab 2: About ===
with tab2:
    st.header("About")
    st.markdown("""
    This app uses a Deep Learning model trained on symptom-disease relationships to predict the most likely diagnosis.

    - 💡 132 binary symptoms
    - 🩺 41 disease classes
    - 🔍 Evaluated on test, noisy & cross-validation data
    - 🧠 Uses Keras ANN with ReLU activations
    - 🌐 Made with Streamlit

    **Built by:** Neel Chauhan
    """)

# === Tab 3: Model Info ===
with tab3:
    st.header("Model Architecture")
    st.markdown("""
    **Model Overview:**
    - Input Layer: 132 nodes
    - Dense(128) + ReLU
    - Dense(64) + ReLU
    - Output: Softmax (41 classes)

    **Training Setup:**
    - Epochs: 20
    - Optimizer: Adam
    - Loss: Sparse Categorical Crossentropy

    **Evaluation Accuracy:**
    - Clean: ✅ 100%
    - Noisy: ✅ ~99.9%
    - 5-Fold CV: ✅ ~99.98%
    """)

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Neel Chauhan | Powered by Streamlit, TensorFlow, and Lottie</p>", unsafe_allow_html=True)
