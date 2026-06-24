import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #0f1117;}
.block-container {
    background-color: rgba(15, 17, 23, 0.95);
    border-radius: 15px;
    padding: 2rem;
    border: 1px solid rgba(255, 0, 0, 0.1);
}
.fraud {color: #F44336; font-size: 28px; font-weight: bold; text-align: center;}
.legit {color: #4CAF50; font-size: 28px; font-weight: bold; text-align: center;}
.risk-high {background: linear-gradient(135deg, #b71c1c, #f44336); padding: 15px; border-radius: 10px; text-align: center; color: white;}
.risk-low {background: linear-gradient(135deg, #1b5e20, #4CAF50); padding: 15px; border-radius: 10px; text-align: center; color: white;}
h1 {color: #FF5252; text-align: center;}
.stButton > button {
    background: linear-gradient(135deg, #b71c1c, #FF5252);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    padding: 10px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(244, 67, 54, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Load model once at top
model = joblib.load('models/fraud_model.pkl')

# Header
st.markdown("<h1>🔍 AI Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888'>Real-time transaction fraud detection powered by Random Forest ML</p>", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("This system uses a **Random Forest** model trained on **284,807 transactions** to detect fraudulent activity in real time.")
    st.divider()
    st.markdown("### 📊 Model Performance")
    st.metric("F1 Score", "0.84")
    st.metric("ROC-AUC", "0.918")
    st.metric("Precision", "85%")
    st.metric("Recall", "84%")
    st.divider()
    st.markdown("### ⚠️ Threshold")
    st.markdown("Probability > **70%** = HIGH RISK")
    st.markdown("Probability < **70%** = LOW RISK")

# Main content
st.markdown("### 💳 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Transaction Info**")
    time = st.number_input("Time (seconds)", value=0.0)
    amount = st.number_input("Amount ($)", value=0.0, min_value=0.0)

with col2:
    st.markdown("**Quick Test**")
    test_type = st.selectbox(
        "Load sample transaction",
        ["Custom Input", "Sample Legitimate", "Sample Fraud"]
    )

# Sample transactions
if test_type == "Sample Legitimate":
    sample = {
        "Time": 0.0, "Amount": 149.62,
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
        "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
        "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
        "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
    }
elif test_type == "Sample Fraud":
    sample = {
        "Time": 406.0, "Amount": 0.00,
        "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99,
        "V5": -0.52, "V6": -1.43, "V7": -2.77, "V8": -2.77,
        "V9": -0.34, "V10": -2.71, "V11": 1.84, "V12": -5.01,
        "V13": -2.68, "V14": -2.65, "V15": -0.09, "V16": -1.40,
        "V17": -2.94, "V18": -3.00, "V19": -0.60, "V20": 1.70,
        "V21": 0.56, "V22": -0.35, "V23": -0.47, "V24": 0.21,
        "V25": 0.77, "V26": 0.91, "V27": -0.69, "V28": -0.06
    }
else:
    sample = None

# V features hidden
v_values = {}
if sample:
    for i in range(1, 29):
        v_values[f"V{i}"] = sample[f"V{i}"]
else:
    for i in range(1, 29):
        v_values[f"V{i}"] = 0.0

st.divider()

# Predict button
if st.button("🔍 Detect Fraud", use_container_width=True):
    payload = {
        "Time": sample["Time"] if sample else time,
        "Amount": sample["Amount"] if sample else amount,
        **v_values
    }

    with st.spinner("🤖 Analyzing transaction..."):
        columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
           'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
           'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
           'V28', 'Amount']
        data = pd.DataFrame([payload])[columns]
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        result = {
            "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": round(float(probability) * 100, 2),
            "status": "HIGH RISK" if probability > 0.7 else "LOW RISK"
        }

        st.divider()
        st.markdown("### 📋 Detection Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            if result["prediction"] == "FRAUD":
                st.markdown(f"<p class='fraud'>🚨 {result['prediction']}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='legit'>✅ {result['prediction']}</p>", unsafe_allow_html=True)

        with col2:
            st.metric("Fraud Probability", f"{result['fraud_probability']}%")

        with col3:
            if result["status"] == "HIGH RISK":
                st.markdown(f"<div class='risk-high'>⚠️ {result['status']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='risk-low'>✅ {result['status']}</div>", unsafe_allow_html=True)

        st.progress(result["fraud_probability"] / 100)

        if result["prediction"] == "FRAUD":
            st.error("🚨 This transaction has been flagged as potentially fraudulent. Immediate review recommended.")
        else:
            st.success("✅ This transaction appears legitimate. No action required.")

st.divider()
st.markdown("<p style='text-align:center;color:#555;font-size:12px'>Built with Random Forest · FastAPI · Streamlit | Trained on 284,807 transactions</p>", unsafe_allow_html=True)