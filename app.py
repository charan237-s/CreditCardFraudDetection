import streamlit as st
import pandas as pd
import joblib

# ===========================
# PAGE CONFIG
# ===========================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Machine Learning Project using Random Forest")

# ===========================
# LOAD MODEL
# ===========================

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===========================
# SIDEBAR
# ===========================

st.sidebar.header("Transaction Details")

# Dataset columns (without Class)
columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"
]

values = []

for col in columns:
    value = st.sidebar.number_input(col, value=0.0)
    values.append(value)

# ===========================
# PREDICTION
# ===========================

if st.sidebar.button("Predict"):

    input_df = pd.DataFrame([values], columns=columns)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("✅ Genuine Transaction")
    else:
        st.error("❌ Fraud Transaction")

    st.write(f"### Genuine Probability : {probability[0]*100:.2f}%")
    st.write(f"### Fraud Probability : {probability[1]*100:.2f}%")