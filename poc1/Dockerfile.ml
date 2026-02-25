FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias de sistema necesarias para clientes S3
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Instalamos las librerías de Python
RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    pandas \
    boto3 \
    psycopg2-binary


CMD ["tail", "-f", "/dev/null"]
