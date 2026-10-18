from fastapi import FastAPI
import pandas as pd
import mlflow

app = FastAPI()

# Correct MLflow model path
model_uri = "mlruns/2/models/m-9dbd09f797a4466cbf8c3644d7fa6313/artifacts"
model = mlflow.sklearn.load_model(model_uri)

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return {"prediction": int(preds[0])}
