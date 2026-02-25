from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import os
import logging
from mlflow import MlflowClient

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ Silenciar logs verbosos de boto3
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

app = FastAPI(title="MLOps Toy API")

MODEL_NAME = "CarroModel"
MODEL_URI = f"models:/{MODEL_NAME}@production"

model = None

@app.on_event("startup")
async def load_model():
    global model
    
    # ✅ Verificar variables de entorno
    logger.info("=" * 60)
    logger.info("🔧 Configuración de Entorno:")
    logger.info(f"  MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    logger.info(f"  MLFLOW_S3_ENDPOINT_URL: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
    logger.info(f"  AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
    logger.info(f"  AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")
    logger.info("=" * 60)
    
    #  Test 1: Verificar conectividad con MLflow Server
    try:
        import requests
        mlflow_url = os.getenv("MLFLOW_TRACKING_URI")
        health_resp = requests.get(f"{mlflow_url}/health", timeout=5)
        logger.info(f" MLflow Server Health: {health_resp.status_code}")
    except Exception as e:
        logger.error(f" MLflow Server no responde: {e}")
        return
    
    #  Test 2: Verificar conectividad con MinIO
    try:
        import boto3
        from botocore.client import Config
        
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
            config=Config(signature_version='s3v4'),
            verify=False  # Desactivar SSL
        )
        
        buckets = s3_client.list_buckets()
        logger.info(f" MinIO accesible. Buckets: {[b['Name'] for b in buckets['Buckets']]}")
        
        # Listar objetos en el bucket mlflow
        try:
            objects = s3_client.list_objects_v2(Bucket='mlflow', MaxKeys=5)
            if 'Contents' in objects:
                logger.info(f" Archivos en bucket 'mlflow': {len(objects['Contents'])} objetos")
                for obj in objects.get('Contents', [])[:3]:
                    logger.info(f"   - {obj['Key']}")
            else:
                logger.warning("  Bucket 'mlflow' está vacío")
        except Exception as list_err:
            logger.error(f" Error listando bucket: {list_err}")
            
    except Exception as e:
        logger.error(f" MinIO no accesible: {e}")
        return
    
    #  Test 3: Cargar el modelo
    try:
        logger.info(f" Intentando cargar modelo: {MODEL_URI}")
        
        # Forzar configuración de S3 para MLflow
        os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
        
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info(" ------------------- ¡Modelo cargado exitosamente!")
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        logger.info(f" Versión del modelo: {model_version.version}")
        
    except Exception as e:
        logger.error(f" ERROR al cargar modelo: {str(e)}", exc_info=True)
        
        #  Diagnóstico adicional
        try:
            
            client = MlflowClient()
            
            # Intentar obtener metadata del modelo
            model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
            logger.info(f" Versión del modelo: {model_version.version}")
            logger.info(f" Run ID: {model_version.run_id}")
            logger.info(f" Source: {model_version.source}")
            
            # Obtener URI del artefacto
            run = client.get_run(model_version.run_id)
            artifact_uri = run.info.artifact_uri
            logger.info(f" Artifact URI: {artifact_uri}")
            
        except Exception as diag_err:
            logger.error(f" Error en diagnóstico: {diag_err}")
        
        model = None

@app.get("/")
def health_check():
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "s3_endpoint": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        "model version" : f"Version del modelo: {model_version.version}"
    }

@app.get("/predict")
def predict():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        dummy_input = [[1, 2]]
        prediction = model.predict(dummy_input)
        
        return {
            "input": dummy_input,
            "prediction": int(prediction[0]),
            "model": MODEL_NAME
        }
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/s3-test")
def test_s3_connection():
    """Endpoint de debug para probar conectividad S3"""
    try:
        import boto3
        from botocore.client import Config
        
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='us-east-1',
            config=Config(signature_version='s3v4'),
            verify=False
        )
        
        buckets = s3_client.list_buckets()
        objects = s3_client.list_objects_v2(Bucket='mlflow', MaxKeys=10)
        
        return {
            "buckets": [b['Name'] for b in buckets['Buckets']],
            "mlflow_objects": [obj['Key'] for obj in objects.get('Contents', [])]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
