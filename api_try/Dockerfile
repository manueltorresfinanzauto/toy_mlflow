FROM python:3.9-slim

WORKDIR /app

# Instalamos dependencias de sistema necesarias para clientes S3
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Instalamos las librerías de Python
RUN pip install fastapi uvicorn mlflow scikit-learn pandas boto3

COPY main.py .

# Exponemos el puerto de FastAPI
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
