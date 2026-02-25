from fastapi import FastAPI, HTTPException
import logging
import requests
import json
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLOps Toy API")

# Usamos variables de entorno para que el Docker sea flexible
MLFLOW_TRACKING_URL = os.getenv("MLFLOW_TRACKING_URL", "http://mlflow_proxy:5000/api/2.0/mlflow")
MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_URL", "http://ml_api_service:5001/invocations")

@app.get("/")
def health_check():
    MODEL_NAME = "RF_test_1"
    ALIAS = "production"
    
    try:
        endpoint = f"{MLFLOW_TRACKING_URL}/registered-models/alias?name={MODEL_NAME}&alias={ALIAS}"
        response = requests.get(endpoint, timeout=5)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Error MLflow: {response.text}")

        return {
            "status": "success",
            "mlflow_data": response.json().get("model_version", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict():
    # Cargar datos y preparar un pequeño ejemplo
    iris = load_iris()
    # Convertimos a DataFrame para usar .to_dict(orient='split') que es lo que pide MLflow
    df_test = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Tomamos las primeras 2 filas como ejemplo
    sample = df_test.head(2)
    
    payload = {
        "dataframe_split": sample.to_dict(orient='split')
    }

    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(MLFLOW_SERVE_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return {
            "prediction": response.json(),
            "model_info": "Consultado desde MLflow Serving"
        }
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))