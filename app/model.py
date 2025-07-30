import pickle
import pandas as pd
from app.schemas import PatientData

with open("model/model.pkl", "rb") as f:
    model, scaler, feature_names = pickle.load(f)

categorical_cols = ["BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"]

def preprocess_input(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0 

    df = df[feature_names]  # Ordenar columnas
    df_scaled = scaler.transform(df)
    return df_scaled

def predict_risk(data: PatientData) -> float:
    X = preprocess_input(data)
    proba = model.predict_proba(X)[0][1]  # Probabilidad de clase 1 (riesgo)
    return round(proba, 4)