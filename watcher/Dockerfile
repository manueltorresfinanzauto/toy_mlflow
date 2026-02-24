FROM python:3.9-slim

# Evitar buffering para logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Instalar Docker CLI para ejecutar comandos en otros contenedores
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary \
    boto3

# Crear directorio de trabajo
WORKDIR /app

# Copiar el script
COPY watcher.py .

# Ejecutar el watcher
CMD ["python", "watcher.py"]
