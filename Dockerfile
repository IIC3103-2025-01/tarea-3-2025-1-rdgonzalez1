# 1. Construye el frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# 2. Construye el backend + copia el build estático
FROM python:3.10-slim

# variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia código backend
COPY src/ ./src
COPY data/docs/ ./docs
COPY embeddings/ .

# copia el build de React desde la etapa 1
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# expone el puerto
EXPOSE ${PORT}

# comando de arranque
CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080"]

RUN mkdir -p /app/embeddings
RUN chmod -R 777 /app/embeddings

