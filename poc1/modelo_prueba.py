import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
import os
import boto3
from botocore.config import Config

try:
    s3_test = boto3.client(
        's3',
        endpoint_url='http://mlflow-s3:9000', # Escrito a mano aquí
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        config=Config(signature_version='s3v4')
    )
    s3_test.list_buckets()
    print("✅ Conexión manual a S3 exitosa")
except Exception as e:
    print(f"❌ Error en conexión manual: {e}")
print(list(os.environ["MLFLOW_S3_ENDPOINT_URL"]))
for var in ["MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
    if var in os.environ:
        os.environ[var] = os.environ[var].replace('\r', '').strip()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow_proxy:5000"))
mlflow.set_experiment("RF_poc_2")

print(f"DEBUG FINAL: '{os.environ['MLFLOW_S3_ENDPOINT_URL']}'")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# 3. ACTIVAR AUTOLOG: Esto capturará los parámetros de GridSearch automáticamente
mlflow.sklearn.autolog(log_models=False)

# Definimos el espacio de búsqueda
param_grid = {
    'n_estimators': [75, 100, 150],
    'max_depth': [ 6, 10, 15],
    'criterion': ['gini', 'entropy']
}


mlflow.sklearn.autolog(log_models=False)


# ✅ Registrar modelo
model_name = "CarroModel"

# 4. Entrenamiento con GridSearchCV
with mlflow.start_run(run_name="RandomForest_GridSearch") as parent_run:
    # Configuramos el GridSearch
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 5. Métricas adicionales sobre el conjunto de TEST (con el mejor modelo)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    signature = infer_signature(X_test, y_pred)
    print(f"Inferred signature:\n{signature}")
    
    # Calculamos métricas manuales para el log
    metrics = {
        "best_cv_score": grid_search.best_score_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, average='weighted'),
        "test_recall": recall_score(y_test, y_pred, average='weighted'),
        "test_f1": f1_score(y_test, y_pred, average='weighted'),
        "modelo_size_gb" : len(pickle.dumps(best_model)) / (1024**3)
    }
    
    # Log de métricas extras que el autolog no captura por defecto en test set
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(best_model, "modelo_final", signature=signature, input_example=X_test[:2])
    
    print(f"Mejor Accuracy en CV: {grid_search.best_score_:.4f}")
    print(f"Accuracy en Test final: {metrics['test_accuracy']:.4f}")
    print(f"ID de la corrida: {parent_run.info.run_id}")

# ✅ Asignar alias "production"
# from mlflow import MlflowClient
# client = MlflowClient()

# # Obtener la última versión registrada
# versions = client.search_model_versions(f"name='{model_name}'")
# latest_version = max([int(v.version) for v in versions])

# # Asignar alias
# client.set_registered_model_alias(
#     name=model_name,
#     alias="production",
#     version=latest_version
# )

# print(f"✅ Alias 'production' asignado a versión {latest_version}")
