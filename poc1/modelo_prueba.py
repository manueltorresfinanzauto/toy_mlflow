import mlflow
from mlflow.tracking import MlflowClient
import os
import time
import subprocess
from datetime import datetime

# --- Configuración ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_proxy:5000")
MODEL_NAME = "CarroModel"
ALIAS = "production"
CONTAINER_TO_RESTART = "api_mlops_test"

# Configuración para GC
DB_URI = "postgresql://mlflow:password@mlflow_db:5432/mlflowdb"
S3_DEST = "s3://mlflow/"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_mlflow_gc():
    print(f"[{get_now()}] 🧹 Ejecutando MLflow Garbage Collector...")
    # AGREGAMOS --tracking-uri al comando para evitar el error de "Tracking URL not set"
    cmd = [
        "docker", "exec", "mlflow_server", 
        "mlflow", "gc",
        "--tracking-uri", TRACKING_URI,
        "--backend-store-uri", DB_URI,
        "--artifacts-destination", S3_DEST
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[{get_now()}] ✨ Espacio en MinIO liberado físicamente.")
    else:
        print(f"[{get_now()}] ❌ ERROR en GC: {result.stderr}")

def cleanup_unaliased_versions():
    print(f"[{get_now()}] 🔍 Buscando versiones sin alias...")
    try:
        # Buscamos versiones
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        deleted_any = False
        
        for v in versions:
            # BLINDAJE: Consultamos la versión individualmente para asegurar que traiga los ALIASES
            full_version_info = client.get_model_version(MODEL_NAME, v.version)
            
            # Si la lista de alias está vacía, es seguro borrar
            if not full_version_info.aliases:
                print(f"[{get_now()}] 🗑️ Borrando registro de Versión {v.version} (Confirmado: Sin alias)")
                client.delete_model_version(name=MODEL_NAME, version=v.version)
                deleted_any = True
            else:
                print(f"[{get_now()}] ✅ Manteniendo Versión {v.version} (Tiene alias: {full_version_info.aliases})")
        
        if deleted_any:
            run_mlflow_gc()
            
    except Exception as e:
        print(f"[{get_now()}] ❌ ERROR en limpieza: {e}")

def get_current_version():
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return alias_info.version
    except Exception:
        return None

# --- Bucle Principal ---
print(f"[{get_now()}] 🚀 WATCHER RECONFIGURADO")
last_version = get_current_version()

while True:
    try:
        current_version = get_current_version()
        
        if current_version != last_version and current_version is not None:
            print(f"[{get_now()}] 🔔 CAMBIO DETECTADO: v{last_version} -> v{current_version}")
            
            # 1. Reiniciar API
            subprocess.run(["docker", "restart", CONTAINER_TO_RESTART])
            
            # 2. Limpiar
            time.sleep(5)
            cleanup_unaliased_versions()
            
            last_version = current_version
        else:
            print(f"[{get_now()}] 🔍 Verificando... v{last_version}")

        time.sleep(20) 
    except Exception as e:
        print(f"[{get_now()}] 💥 Error: {e}")
        time.sleep(10)