# API de predicción de supervivencia del Titanic 🚢

Team Challenge Final — Despliegue de un modelo de Machine Learning como API
REST accesible públicamente, con **FastAPI**.

## Índice
- [Estructura del proyecto](#estructura-del-proyecto)
- [Modelo](#modelo)
- [Probar en local](#probar-en-local)
- [Endpoints](#endpoints)
- [Desplegar en Render](#desplegar-en-render)
- [Redespliegue en directo (tercer endpoint)](#redespliegue-en-directo-tercer-endpoint)
- [Git Flow](#git-flow)

## Estructura del proyecto

```
.
├── app.py                  # API FastAPI: landing page + endpoints
├── train_model.py          # Entrena el pipeline y genera modelo_titanic.joblib
├── modelo_titanic.joblib   # Modelo ya entrenado (Pipeline de sklearn)
├── titanic_train.csv       # Dataset de entrenamiento
├── test_api.py             # Pruebas locales con requests antes de desplegar
├── requirements.txt        # Dependencias exactas
├── Procfile                # Comando de arranque (Render/Railway/Heroku-like)
├── render.yaml             # Blueprint de Render (despliegue "as code")
├── .env.example             # Ejemplo de variables de entorno opcionales
└── .gitignore
```

## Modelo

`train_model.py` entrena un `RandomForestClassifier` dentro de un `Pipeline`
de scikit-learn con 4 variables: `Pclass`, `Sex`, `Age`, `Fare`. El
preprocesado (imputación de nulos, escalado y one-hot de `Sex`) vive dentro
del propio pipeline, así que el `.joblib` que se guarda ya incluye todo —
la API solo necesita cargarlo y llamar a `.predict()` / `.predict_proba()`.

Para regenerar el modelo desde cero:

```bash
python train_model.py
```

## Probar en local

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app:app --reload
```

Abre **http://127.0.0.1:8000/docs** para la documentación interactiva
(Swagger), generada automáticamente por FastAPI.

En otra terminal, con el servidor corriendo:

```bash
python test_api.py
```

Esto lanza varias peticiones (`GET /`, `GET /predict_get`, `POST /predict`,
y dos casos de datos inválidos) para comprobar que todo responde bien antes
de tocar producción.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Landing page: explica cómo usar el resto de endpoints |
| `POST` | `/predict` | JSON en el body `{Pclass, Sex, Age, Fare}` → predicción |
| `GET` | `/predict_get` | Query string `?Pclass=1&Sex=female&Age=29&Fare=100` → predicción (pensado para `requests.get`) |
| `GET` | `/docs` | Documentación Swagger interactiva |

Ejemplo con `requests` en Python:

```python
import requests

r = requests.get(
    "https://TU-APP.onrender.com/predict_get",
    params={"Pclass": 1, "Sex": "female", "Age": 29, "Fare": 100},
)
print(r.json())
# {'survived': 1, 'probabilidad_supervivencia': 0.87}
```

**Manejo de errores:** al ir todo tipado con Pydantic (`Pclass` entre 1 y 3,
`Sex` solo `"male"`/`"female"`, `Age` y `Fare` positivas), si mandas un JSON
mal formado o con un campo fuera de rango, FastAPI responde automáticamente
con un `422` y el detalle de qué campo falló — nunca un `500` críptico.

## Desplegar en Render

1. Sube este repositorio a GitHub (ver sección [Git Flow](#git-flow) más
   abajo si quieres seguir el flujo de ramas recomendado).
2. Entra en [render.com](https://render.com) → **New** → **Web Service**.
3. Conecta tu repositorio de GitHub.
4. Configura:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
   
   (Si usas el `render.yaml` incluido, Render puede leer esta configuración
   automáticamente al crear el servicio como "Blueprint".)
5. Deploy. Tras el build, Render te da una URL pública del tipo
   `https://titanic-api-xxxx.onrender.com`.
6. Prueba `GET https://tu-url.onrender.com/` y `GET .../predict_get?...`
   desde el navegador o con `requests.get`.

> ⚠️ **El plan gratuito de Render se "duerme"** tras 15 minutos de
> inactividad y tarda 30-60s en despertar con la primera petición. Si vas a
> hacer una demo en directo, abre tu propia URL un par de minutos antes.
> Truco: usa [cron-job.org](https://cron-job.org) (gratis) para que haga una
> petición a tu URL cada 10-14 minutos y no se duerma — actívalo el día
> antes de la presentación y desactívalo después.

### Variables de entorno

El proyecto no necesita ninguna variable obligatoria para funcionar, pero
`MODEL_PATH` y `DATA_PATH` (ver `train_model.py`) se pueden sobreescribir
por variable de entorno en vez de tocar el código — útil si algún día
quieres apuntar a otro modelo/dataset sin hacer un nuevo despliegue. Se
configuran en Render en **Settings → Environment**, nunca en el repo (de
ahí el `.env.example` en vez de un `.env` real).

## Redespliegue en directo (tercer endpoint)

En `app.py` hay un tercer endpoint, `/predict_batch`, ya escrito pero
**comentado a propósito**, para la demo de "redespliegue en directo":

```bash
git switch -c feature/predict-batch
# descomentar el bloque de /predict_batch en app.py
git add app.py
git commit -m "feat: add batch prediction endpoint"
git push -u origin feature/predict-batch
# abrir Pull Request feature/predict-batch -> main en GitHub, y mergear
```

En cuanto el merge llega a `main`, Render detecta el push y redespliega
automáticamente — sin tocar nada en el propio Render.

## Git Flow

Flujo de ramas recomendado (no bloqueante, pero buena práctica):

| Rama | Propósito |
|---|---|
| `main` | Código desplegado. Solo recibe merges vía Pull Request. |
| `develop` | Integración de features antes de pasar a producción. |
| `feature/nombre` | Una funcionalidad nueva. Sale de `develop`, vuelve a `develop`. |
| `fix/nombre` | Corrección de un bug. Mismo origen/destino que `feature`. |

```bash
git switch develop
git pull origin develop
git switch -c feature/mi-funcionalidad

# ... trabajar, con commits tipo Conventional Commits (feat:, fix:, docs:) ...

git add .
git commit -m "feat: descripción breve del cambio"
git push -u origin feature/mi-funcionalidad
# -> Pull Request en GitHub: base develop <- compare feature/mi-funcionalidad
```
