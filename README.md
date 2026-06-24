# 🔍 AI Fraud Detection System

Real-time credit card fraud detection powered by Random Forest ML.

## 🚀 What it does
- Detects fraudulent transactions in real time
- Trained on 284,807 real credit card transactions
- Handles class imbalance using SMOTE
- Shows fraud probability and risk level

## 📊 Model Performance
- F1 Score: 0.84
- ROC-AUC: 0.918
- Precision: 85%
- Recall: 84%

## 🛠️ Tech Stack
- **Scikit-learn** — Logistic Regression, Decision Tree, Random Forest
- **XGBoost** — Gradient boosting classifier
- **SMOTE** — Class balancing
- **Streamlit** — Interactive UI
- **Joblib** — Model persistence

## ⚙️ How to Run
1. Clone the repository
2. Install: `pip install -r requirements.txt`
3. Train model: `python train.py`
4. Run app: `python -m streamlit run app.py`

## 📁 Dataset
Kaggle Credit Card Fraud Detection — 284,807 transactions, 0.17% fraud rate
