from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.model import predict_risk, PatientData

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict_form(
    request: Request,
    Age: int = Form(...),
    Salt_Intake: float = Form(...),
    Stress_Score: float = Form(...),
    BP_History: str = Form(...),
    Sleep_Duration: float = Form(...),
    BMI: float = Form(...),
    Medication: str = Form(...),
    Family_History: str = Form(...),
    Exercise_Level: str = Form(...),
    Smoking_Status: str = Form(...)
):
    data = PatientData(
        Age=Age,
        Salt_Intake=Salt_Intake,
        Stress_Score=Stress_Score,
        BP_History=BP_History,
        Sleep_Duration=Sleep_Duration,
        BMI=BMI,
        Medication=Medication,
        Family_History=Family_History,
        Exercise_Level=Exercise_Level,
        Smoking_Status=Smoking_Status
    )
    result = predict_risk(data)
    return templates.TemplateResponse("form.html", {"request": request, "prediction": result})