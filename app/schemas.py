from pydantic import BaseModel

class PatientData(BaseModel):
    Age: int
    Salt_Intake: float
    Stress_Score: float
    BP_History: str
    Sleep_Duration: float
    BMI: float
    Medication: str
    Family_History: str
    Exercise_Level: str
    Smoking_Status: str

class PredictionResponse(BaseModel):
    hypertension_risk: float