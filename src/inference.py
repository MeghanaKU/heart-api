import joblib
import pandas as pd

MODEL_PATH = "models/final_model.pkl"

# Columns used during training
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

def load_model():
    return joblib.load(MODEL_PATH)

def predict_single(sample: dict):
    model = load_model()

    # Create DataFrame with correct column order
    df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].max()

    return int(pred), float(proba)
