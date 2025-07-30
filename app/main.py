from fastapi import FastAPI
from app.schemas import PatientData, PredictionResponse
from app.model import predict_risk

app = FastAPI(title="Hypertension Risk Inference API")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    risk = predict_risk(data)
    return PredictionResponse(hypertension_risk=risk)

from fastapi.middleware.cors import CORSMiddleware