FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias
RUN pip install --no-cache-dir fastapi uvicorn requests pandas scikit-learn

# Copiamos el código
COPY main.py .

# Exponemos el puerto de FastAPI
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
