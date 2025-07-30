# Hypertension Risk ML Inference Service

Este microservicio realiza inferencias de riesgo de hipertensión a partir de datos clínicos usando un modelo entrenado con Random Forest y servido con FastAPI.

## Estructura
- `app/`: Código FastAPI (API, esquemas, modelo)
- `model/`: Archivos del modelo (`.pkl`) generados por `train_model.py` y Dataset original (no se sube a GitHub)