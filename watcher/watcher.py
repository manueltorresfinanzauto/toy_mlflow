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

# Configuración para GC (Garbage Collector)
DB_URI = "postgresql://mlflow:password@mlflow_db:5432/mlflowdb"
S3_DEST = "s3://mlflow/"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_version():
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return alias_info.version
    except Exception:
        return None

def run_mlflow_gc():
    """Ejecuta el GC dentro del contenedor de MLflow para borrar archivos físicos en MinIO"""
    print(f"[{get_now()}] 🧹 Ejecutando MLflow Garbage Collector en MinIO...")
    cmd = [
        "docker", "exec", "mlflow_server", 
        "mlflow", "gc",
        "--backend-store-uri", DB_URI,
        "--artifacts-destination", S3_DEST
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[{get_now()}] ✨ Espacio en MinIO liberado físicamente.")
    else:
        print(f"[{get_now()}] ❌ ERROR en GC: {result.stderr}")

def cleanup_unaliased_versions():
    """Borra de la base de datos las versiones que ya no tienen ningún alias"""
    print(f"[{get_now()}] 🔍 Buscando versiones huérfanas (sin alias)...")
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        deleted_any = False
        
        for v in versions:
            # Si no tiene alias, la borramos
            if not v.aliases:
                print(f"[{get_now()}] 🗑️ Borrando registro de Versión {v.version} (Sin alias)")
                client.delete_model_version(name=MODEL_NAME, version=v.version)
                deleted_any = True
        
        if deleted_any:
            # Si borramos registros, disparamos el GC para limpiar los archivos de 8GB
            run_mlflow_gc()
        else:
            print(f"[{get_now()}] ✅ No se encontraron versiones para limpiar.")
            
    except Exception as e:
        print(f"[{get_now()}] ❌ ERROR durante la limpieza de versiones: {e}")

# --- Inicio del Script ---
print(f"[{get_now()}] 🚀 WATCHER INICIADO")
print(f"[{get_now()}] 🎯 Monitoreando: {MODEL_NAME} @ {ALIAS}")
print(f"[{get_now()}] 🔌 Conectado a: {TRACKING_URI}")

last_version = get_current_version()
print(f"[{get_now()}] 📍 Versión actual en producción: {last_version}")

while True:
    try:
        current_version = get_current_version()
        
        if current_version is None:
            print(f"[{get_now()}] ⚠️ Advertencia: No se encontró versión con alias '{ALIAS}'.")
        
        elif current_version != last_version:
            print(f"[{get_now()}] 🔔 ¡CAMBIO DE MODELO DETECTADO!")
            print(f"[{get_now()}] 📉 Antigua: {last_version} | 📈 Nueva: {current_version}")
            print(f"[{get_now()}] 🔄 Reiniciando {CONTAINER_TO_RESTART} para liberar 8GB de RAM...")
            
            # Ejecutar reinicio de la API
            start_time = time.time()
            result = subprocess.run(["docker", "restart", CONTAINER_TO_RESTART], capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"[{get_now()}] ✅ Reinicio exitoso en {round(end_time - start_time, 2)} segundos.")
                
                # Después del reinicio, procedemos a limpiar el desorden viejo
                time.sleep(5) # Pausa técnica
                cleanup_unaliased_versions()
                
                last_version = current_version
            else:
                print(f"[{get_now()}] ❌ ERROR al reiniciar contenedor: {result.stderr}")
        
        else:
            # Tu log de latido (Heartbeat)
            print(f"[{get_now()}] 🔍 Verificando... Sin cambios (Versión actual: {last_version})")

        time.sleep(20) 

    except Exception as e:
        print(f"[{get_now()}] 💥 Error crítico en el bucle: {e}")
        time.sleep(10)