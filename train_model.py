"""Entrena un modelo mínimo de supervivencia del Titanic para la demo de despliegue.

Ejecútalo como script para (re)generar el modelo:
    python train_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PWD = os.path.dirname(os.path.abspath(__file__))

# Rutas configurables por variable de entorno en vez de "a fuego" en el código
# (buena práctica: así Render, o cualquier otro entorno, puede sobreescribirlas
# sin tocar el código fuente).
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(PWD, "titanic_train.csv"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PWD, "modelo_titanic.joblib"))

FEATURES = ["Pclass", "Sex", "Age", "Fare"]
TARGET = "Survived"


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
                ["Age", "Fare"],
            ),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), ["Sex"]),
        ],
        remainder="passthrough",
    )
    return Pipeline([("preprocesado", preprocessing), ("modelo", RandomForestClassifier(random_state=42))])


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> Pipeline:
    """Entrena el pipeline con los datos de `data_path` y lo guarda en `model_path`."""
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    joblib.dump(pipeline, model_path)
    return pipeline


if __name__ == "__main__":
    modelo_entrenado = train()
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(
        "Ejemplo de predicción:",
        modelo_entrenado.predict(pd.DataFrame([{"Pclass": 1, "Sex": "female", "Age": 29, "Fare": 100}])),
    )
