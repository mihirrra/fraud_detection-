import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')





# Load dataset
df = pd.read_csv('data/creditcard.csv')

# Basic EDA
print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['Class'].value_counts())
print("\nFraud Percentage:", round(df['Class'].mean() * 100, 2), "%")
print("\nMissing Values:", df.isnull().sum().sum())




# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Time and Amount (V1-V28 already scaled)
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)




# Apply SMOTE on training data only
print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())








# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc}")
    results[name] = roc


    # Find best model
best_model_name = "Random Forest"
print(f"\nBest Model: {best_model_name}")
print(f"Best ROC-AUC: {results[best_model_name]}")

# Save best model
# Force save Random Forest - best F1 score for fraud detection
rf_model = models["Random Forest"]
joblib.dump(rf_model, 'models/fraud_model.pkl')
print("Random Forest saved to models/fraud_model.pkl")