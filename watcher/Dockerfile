FROM python:3.9-slim

# Instalamos el cliente de Docker para que el contenedor pueda ejecutar 'docker restart'
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir mlflow

# Copiamos el script al contenedor
COPY watcher.py /app/watcher.py

WORKDIR /app

# Ejecutamos el script
CMD ["python", "watcher.py"]