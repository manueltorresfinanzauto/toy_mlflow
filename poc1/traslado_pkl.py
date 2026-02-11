import mlflow
import pickle
import os

# 1. Configuración de conexión (Ajusta si los nombres de tus contenedores son distintos)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000" # MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

mlflow.set_tracking_uri("http://localhost:5000") # MLflow Server

# 2. Cargar tu modelo local
model_path = "./Data/tu_modelo.pkl" # La ruta de tu archivo actual
with open(model_path, 'rb') as f:
    model_obj = pickle.load(f)

# 3. Subirlo a MLflow
model_name = "CarroModel"

with mlflow.start_run(run_name="Carga manual de modelo"):
    # Logeamos el modelo en el storage (MinIO)
    # Usamos mlflow.sklearn si es de scikit-learn, o pyfunc para cualquier pickle
    mlflow.sklearn.log_model(
        sk_model=model_obj, 
        artifact_path="modelo_pkl",
        registered_model_name=model_name
    )
    print(f"Modelo subido y registrado como: {model_name}")