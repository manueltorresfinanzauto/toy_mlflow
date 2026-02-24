import mlflow
from mlflow.tracking import MlflowClient
import os
import time
import subprocess
from datetime import datetime

# --- Configuración ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_proxy:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "CarroModel")
ALIAS = os.getenv("ALIAS", "production")
CONTAINER_TO_RESTART = os.getenv("CONTAINER_TO_RESTART", "api_mlops_test")

# Configuración para GC (Garbage Collector)
DB_URI = os.getenv("DB_URI", "postgresql://mlflow:password@mlflow_db:5432/mlflowdb")
S3_DEST = "s3://mlflow/"

# Credenciales AWS/MinIO
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3.local:9000")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

def get_now():
    """Timestamp formateado para logs"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_version():
    """Obtiene la versión actual con el alias de producción"""
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return alias_info.version
    except Exception:
        return None

def delete_orphan_runs(version):
    """
    CRÍTICO: Elimina el Run asociado a una versión.
    MLflow NO borrará archivos físicos si el Run está activo.
    """
    try:
        version_detail = client.get_model_version(MODEL_NAME, version)
        run_id = version_detail.run_id
        
        if run_id:
            print(f"[{get_now()}] 🎯 Eliminando Run {run_id} asociado a versión {version}...")
            client.delete_run(run_id)
            print(f"[{get_now()}] ✅ Run {run_id} eliminado exitosamente")
            return True
        else:
            print(f"[{get_now()}] ⚠️ Versión {version} no tiene Run asociado")
            return False
            
    except Exception as e:
        print(f"[{get_now()}] ❌ Error eliminando Run de versión {version}: {e}")
        return False

def run_mlflow_gc():
    """
    Ejecuta el Garbage Collector de MLflow DENTRO del contenedor mlflow_server.
    Esto es crucial porque solo así tiene acceso directo a MinIO.
    
    IMPORTANTE: Usa --older-than 0s para evitar el periodo de gracia de 30 días.
    """
    print(f"[{get_now()}] 🧹 Iniciando MLflow Garbage Collector...")
    print(f"[{get_now()}] 📍 Target: MinIO (Liberando archivos de 8GB)")
    
    # OPCIÓN 1: Usar --backend-store-uri directamente (más confiable)
    cmd = [
        "docker", "exec",
        "-e", f"AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY}",
        "-e", f"AWS_SECRET_ACCESS_KEY={AWS_SECRET_KEY}",
        "-e", f"MLFLOW_S3_ENDPOINT_URL={S3_ENDPOINT}",
        "-e", "AWS_DEFAULT_REGION=us-east-1",
        "mlflow_server",
        "mlflow", "gc",
        "--backend-store-uri", DB_URI,
        "--artifacts-destination", S3_DEST,
        "--older-than", "0s"
    ]
    
    print(f"[{get_now()}] 🔧 Ejecutando comando GC...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"[{get_now()}] ✨ ¡GC EXITOSO! Archivos físicos eliminados de MinIO")
            if result.stdout:
                print(f"[{get_now()}] 📋 Output GC:\n{result.stdout}")
            return True
        else:
            print(f"[{get_now()}] ❌ ERROR en GC (código {result.returncode})")
            print(f"[{get_now()}] 📋 STDERR: {result.stderr}")
            
            # Si falla, intentar OPCIÓN 2: usando MLFLOW_TRACKING_URI
            print(f"[{get_now()}] 🔄 Reintentando con MLFLOW_TRACKING_URI...")
            
            cmd_alt = [
                "docker", "exec",
                "-e", f"MLFLOW_TRACKING_URI={DB_URI}",  # Apuntar directamente a PostgreSQL
                "-e", f"AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY}",
                "-e", f"AWS_SECRET_ACCESS_KEY={AWS_SECRET_KEY}",
                "-e", f"MLFLOW_S3_ENDPOINT_URL={S3_ENDPOINT}",
                "-e", "AWS_DEFAULT_REGION=us-east-1",
                "mlflow_server",
                "mlflow", "gc",
                "--artifacts-destination", S3_DEST,
                "--older-than", "0s"
            ]
            
            result_alt = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=300)
            
            if result_alt.returncode == 0:
                print(f"[{get_now()}] ✨ ¡GC EXITOSO con método alternativo!")
                if result_alt.stdout:
                    print(f"[{get_now()}] 📋 Output GC:\n{result_alt.stdout}")
                return True
            else:
                print(f"[{get_now()}] ❌ Método alternativo también falló")
                print(f"[{get_now()}] 📋 STDERR: {result_alt.stderr}")
                return False
            
    except subprocess.TimeoutExpired:
        print(f"[{get_now()}] ⏱️ TIMEOUT: GC tardó más de 5 minutos")
        return False
    except Exception as e:
        print(f"[{get_now()}] 💥 Excepción al ejecutar GC: {e}")
        return False

def verify_version_has_no_alias(version):
    """
    CRÍTICO: Verifica con get_model_version (NO con search) que 
    la versión realmente no tiene alias antes de borrar.
    
    search_model_versions tiene cache y puede mostrar alias obsoletos.
    """
    try:
        version_detail = client.get_model_version(MODEL_NAME, version)
        has_alias = len(version_detail.aliases) > 0
        
        if has_alias:
            print(f"[{get_now()}] ⚠️ Versión {version} SÍ tiene alias: {version_detail.aliases}")
        
        return not has_alias
        
    except Exception as e:
        print(f"[{get_now()}] ❌ Error verificando versión {version}: {e}")
        return False

def cleanup_unaliased_versions():
    """
    Pipeline completo de limpieza:
    1. Busca versiones sin alias
    2. Verifica con get_model_version (evita falsos positivos)
    3. Elimina el Run asociado (prerequisito para borrado físico)
    4. Elimina el registro de la versión
    5. Ejecuta GC para borrar archivos de 8GB en MinIO
    """
    print(f"[{get_now()}] 🔍 Iniciando limpieza de versiones huérfanas...")
    
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        versions_to_delete = []
        
        # PASO 1: Identificar candidatos
        for v in versions:
            if not v.aliases:  # Primera verificación rápida
                versions_to_delete.append(v.version)
        
        if not versions_to_delete:
            print(f"[{get_now()}] ✅ No hay versiones huérfanas. Sistema limpio.")
            return
        
        print(f"[{get_now()}] 📊 Candidatos a eliminar: {versions_to_delete}")
        
        deleted_count = 0
        runs_deleted = 0
        
        for version in versions_to_delete:
            # PASO 2: Verificación doble con get_model_version
            if not verify_version_has_no_alias(version):
                print(f"[{get_now()}] ⏭️ Saltando versión {version} (tiene alias)")
                continue
            
            # PASO 3: Eliminar el Run (CRÍTICO para borrado físico)
            if delete_orphan_runs(version):
                runs_deleted += 1
                time.sleep(1)  # Pausa técnica para que MLflow procese
            
            # PASO 4: Eliminar registro de versión
            try:
                print(f"[{get_now()}] 🗑️ Eliminando versión {version} del registro...")
                client.delete_model_version(name=MODEL_NAME, version=version)
                deleted_count += 1
                print(f"[{get_now()}] ✅ Versión {version} eliminada del registro")
                time.sleep(1)
                
            except Exception as e:
                print(f"[{get_now()}] ❌ Error eliminando versión {version}: {e}")
        
        # PASO 5: Ejecutar GC para borrado físico en MinIO
        if deleted_count > 0:
            print(f"[{get_now()}] 📊 Resumen: {deleted_count} versiones | {runs_deleted} runs eliminados")
            print(f"[{get_now()}] ⏳ Esperando 3s antes de ejecutar GC...")
            time.sleep(3)
            
            gc_success = run_mlflow_gc()
            
            if gc_success:
                print(f"[{get_now()}] 🎉 LIMPIEZA COMPLETA: {deleted_count} versiones + archivos físicos eliminados")
            else:
                print(f"[{get_now()}] ⚠️ Versiones eliminadas pero GC falló. Archivos físicos pueden persistir.")
        else:
            print(f"[{get_now()}] ℹ️ No se eliminaron versiones en esta ejecución")
            
    except Exception as e:
        print(f"[{get_now()}] 💥 ERROR CRÍTICO durante limpieza: {e}")
        import traceback
        traceback.print_exc()

# --- INICIO DEL WATCHER ---
print("=" * 80)
print(f"[{get_now()}] 🚀 MLFLOW WATCHER INICIADO")
print("=" * 80)
print(f"[{get_now()}] 🎯 Modelo: {MODEL_NAME}")
print(f"[{get_now()}] 🏷️ Alias: {ALIAS}")
print(f"[{get_now()}] 🔌 MLflow: {TRACKING_URI}")
print(f"[{get_now()}] 🗄️ Database: {DB_URI}")
print(f"[{get_now()}] 📦 MinIO: {S3_ENDPOINT}")
print(f"[{get_now()}] 🐳 Contenedor API: {CONTAINER_TO_RESTART}")
print("=" * 80)

last_version = get_current_version()
print(f"[{get_now()}] 📍 Versión actual en producción: {last_version}")

# Loop principal
while True:
    try:
        current_version = get_current_version()
        
        if current_version is None:
            print(f"[{get_now()}] ⚠️ Advertencia: No se encontró versión con alias '{ALIAS}'")
        
        elif current_version != last_version:
            print("\n" + "=" * 80)
            print(f"[{get_now()}] 🔔 ¡¡¡ CAMBIO DE MODELO DETECTADO !!!")
            print("=" * 80)
            print(f"[{get_now()}] 📉 Versión anterior: {last_version}")
            print(f"[{get_now()}] 📈 Versión nueva: {current_version}")
            print(f"[{get_now()}] 🔄 Reiniciando {CONTAINER_TO_RESTART}...")
            
            # Reinicio de la API
            start_time = time.time()
            result = subprocess.run(
                ["docker", "restart", CONTAINER_TO_RESTART],
                capture_output=True,
                text=True
            )
            end_time = time.time()
            
            if result.returncode == 0:
                elapsed = round(end_time - start_time, 2)
                print(f"[{get_now()}] ✅ API reiniciada exitosamente ({elapsed}s)")
                print(f"[{get_now()}] 💾 8GB de RAM liberados")
                
                # Pausa técnica para estabilización
                print(f"[{get_now()}] ⏳ Esperando 5s para estabilización...")
                time.sleep(5)
                
                # Limpieza completa
                cleanup_unaliased_versions()
                
                last_version = current_version
                print("=" * 80 + "\n")
            else:
                print(f"[{get_now()}] ❌ ERROR al reiniciar: {result.stderr}")
        
        else:
            # Heartbeat - log periódico
            print(f"[{get_now()}] 💓 Heartbeat | Versión: {last_version} | Estado: OK")

        time.sleep(20)

    except KeyboardInterrupt:
        print(f"\n[{get_now()}] 🛑 Watcher detenido por usuario")
        break
    except Exception as e:
        print(f"[{get_now()}] 💥 ERROR CRÍTICO en bucle principal: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(10)
