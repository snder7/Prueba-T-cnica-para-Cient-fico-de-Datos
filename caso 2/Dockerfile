# ====================================================================
# DOCKERFILE - STUDENT PERFORMANCE PREDICTION API
# ====================================================================
# Docker image para deployment de la API de predicción de rendimiento
# estudiantil con FastAPI y Machine Learning.
#
# Build: docker build -t student-performance-api .
# Run:   docker run -p 8000:8000 student-performance-api
# ====================================================================

# Imagen base de Python 3.10
FROM python:3.10-slim

# Metadata
LABEL maintainer="Científico de Datos"
LABEL description="API de predicción de rendimiento estudiantil"
LABEL version="1.0.0"

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY models/ ./models/

# Crear directorio para logs
RUN mkdir -p /app/logs

# Exponer puerto de la API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para ejecutar la aplicación
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
