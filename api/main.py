from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict_single

class Patient(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

app = FastAPI()

@app.post("/predict")
def predict(patient: Patient):
    pred, proba = predict_single(patient.dict())
    return {"prediction": pred, "confidence": proba}
