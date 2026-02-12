import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mlflow
import os

# ✅ Datos de entrenamiento
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
model.fit(X, y)

# ✅ Configuración de MLflow (usa localhost porque ejecutas FUERA de Docker)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

mlflow.set_tracking_uri("http://localhost:5000")

# ✅ Registrar modelo
model_name = "CarroModel"

with mlflow.start_run(run_name="Modelo_Toy_v1"):
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # ⚠️ Usa "model" como convención estándar
        registered_model_name=model_name
    )
    mlflow.log_params({"n_estimators": 2, "max_depth": 2})
    mlflow.log_metric("dummy_accuracy", 1.0)
    
    run_id = mlflow.active_run().info.run_id
    print(f"✅ Modelo registrado con Run ID: {run_id}")

# ✅ Asignar alias "production"
from mlflow import MlflowClient
client = MlflowClient()

# Obtener la última versión registrada
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max([int(v.version) for v in versions])

# Asignar alias
client.set_registered_model_alias(
    name=model_name,
    alias="production",
    version=latest_version
)

print(f"✅ Alias 'production' asignado a versión {latest_version}")
