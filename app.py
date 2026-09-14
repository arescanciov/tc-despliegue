"""
API REST con FastAPI — predicción de supervivencia del Titanic.

Probar en local antes de desplegar:
    uvicorn app:app --reload
    # abre http://127.0.0.1:8000/docs para la interfaz Swagger interactiva
    # o ejecuta test_api.py en otra terminal

Endpoints:
    GET  /               -> landing page: cómo usar el resto de endpoints
    POST /predict         -> predicción a partir de un JSON en el body
    GET  /predict_get     -> predicción a partir de una query string (requests.get)
    # POST /predict_batch  -> (comentado a propósito, ver más abajo) predicción por lotes
"""
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal  # noqa: F401 (List se usa al descomentar /predict_batch)
import joblib
import pandas as pd

from train_model import MODEL_PATH

app = FastAPI(title="API de predicción de supervivencia del Titanic")
modelo = joblib.load(MODEL_PATH)


class Pasajero(BaseModel):
    Pclass: int = Field(ge=1, le=3, description="Clase del billete: 1, 2 o 3")
    Sex: Literal["male", "female"]
    Age: float = Field(gt=0, le=120, description="Edad en años")
    Fare: float = Field(ge=0, description="Tarifa pagada")


class Prediccion(BaseModel):
    survived: int
    probabilidad_supervivencia: float


@app.get("/")
def home():
    return {
        "mensaje": "API de predicción de supervivencia del Titanic",
        "endpoints": {
            "/predict": "POST con JSON {Pclass, Sex, Age, Fare} en el body --- devuelve la predicción",
            "/predict_get": "GET con query string ?Pclass=1&Sex=female&Age=29&Fare=100 --- devuelve la predicción",
            "/docs": "documentación interactiva (Swagger)",
        },
    }


@app.post("/predict", response_model=Prediccion)
def predict(pasajero: Pasajero):
    X = pd.DataFrame([pasajero.model_dump()])
    prediccion = int(modelo.predict(X)[0])
    probabilidad = float(modelo.predict_proba(X)[0][1])
    return Prediccion(survived=prediccion, probabilidad_supervivencia=round(probabilidad, 3))


@app.get("/predict_get", response_model=Prediccion)
def predict_get(pasajero: Pasajero = Depends()):
    X = pd.DataFrame([pasajero.model_dump()])
    prediccion = int(modelo.predict(X)[0])
    probabilidad = float(modelo.predict_proba(X)[0][1])
    return Prediccion(survived=prediccion, probabilidad_supervivencia=round(probabilidad, 3))


# ---------------------------------------------------------------------------
# TERCER ENDPOINT (comentado a propósito) — para la demo de "redespliegue
# en directo": predicción por lotes (varios pasajeros en una sola llamada).
#
# En la presentación:
#   1. git switch -c feature/predict-batch
#   2. Descomentar el bloque de abajo
#   3. git add . && git commit -m "feat: add batch prediction endpoint"
#   4. git push -u origin feature/predict-batch  -> abrir Pull Request a main
#   5. Mergear el PR -> Render redespliega solo al detectar el push a main
# ---------------------------------------------------------------------------
# class PrediccionBatch(BaseModel):
#     resultados: List[Prediccion]
#
#
# @app.post("/predict_batch", response_model=PrediccionBatch)
# def predict_batch(pasajeros: List[Pasajero]):
#     X = pd.DataFrame([p.model_dump() for p in pasajeros])
#     predicciones = modelo.predict(X)
#     probabilidades = modelo.predict_proba(X)[:, 1]
#     resultados = [
#         Prediccion(survived=int(p), probabilidad_supervivencia=round(float(prob), 3))
#         for p, prob in zip(predicciones, probabilidades)
#     ]
#     return PrediccionBatch(resultados=resultados)
