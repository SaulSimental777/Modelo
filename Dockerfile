FROM python:3.11-slim

# TensorFlow necesita estas dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero (aprovecha cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el modelo pre-entrenado
COPY . .

# Railway inyecta PORT automáticamente
ENV PORT=3100

EXPOSE $PORT

# Usar gunicorn en producción (no el servidor de desarrollo de Flask)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
