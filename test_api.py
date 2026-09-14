"""
Pruebas rápidas contra la API en local antes de desplegar.

1. En una terminal, arranca el servidor:
       uvicorn app:app --reload
2. En otra terminal, ejecuta este script:
       python test_api.py

También sirve contra la URL ya desplegada: cambia BASE_URL por la de Render.
"""
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_home():
    r = requests.get(f"{BASE_URL}/")
    print("GET /                    ->", r.status_code, r.json())


def test_predict_get():
    params = {"Pclass": 1, "Sex": "female", "Age": 29, "Fare": 100}
    r = requests.get(f"{BASE_URL}/predict_get", params=params)
    print("GET /predict_get         ->", r.status_code, r.json())


def test_predict_post():
    payload = {"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("POST /predict            ->", r.status_code, r.json())


def test_predict_datos_invalidos():
    # Pclass fuera de rango, Sex no permitido, Age negativa -> debe devolver
    # un 422 con el detalle del error, nunca un 500 críptico.
    payload = {"Pclass": 9, "Sex": "otro", "Age": -5, "Fare": 100}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("POST /predict (inválido) ->", r.status_code, r.json())


def test_predict_campo_faltante():
    payload = {"Pclass": 1, "Sex": "female", "Age": 29}  # falta Fare
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("POST /predict (sin Fare) ->", r.status_code, r.json())


if __name__ == "__main__":
    test_home()
    test_predict_get()
    test_predict_post()
    test_predict_datos_invalidos()
    test_predict_campo_faltante()
