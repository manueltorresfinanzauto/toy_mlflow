#!/bin/bash
set -e

echo "--- Iniciando verificación de modelo ---"
echo "Tracking URI: $MLFLOW_TRACKING_URI"

URL="${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/alias?name=RF_test_1&alias=production"
curl -v "${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/alias?name=RF_test_1&alias=production"

until curl -s -f "$URL" > /dev/null 2>&1; do
  echo "Modelo RF_test_1@production no encontrado aún. Reintentando en 10s..."
  sleep 10
done

echo "--- ¡Modelo detectado exitosamente! ---"

exec mlflow models serve --model-uri "models:/RF_test_1@production" --port 5001 --host 0.0.0.0 --no-conda
