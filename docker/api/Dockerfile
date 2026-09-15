# =====================================================================
# DOCKERFILE: FastAPI Backend Inferencia (CRISP-DM Fase 7.6)
# Mantenimiento Predictivo - Área de Helados
# =====================================================================

FROM python:3.12-slim

# Evitar generación de archivos .pyc y habilitar salida de buffer inmediata
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copiar código fuente y artefactos del modelo
COPY services/ /app/services/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY results/ /app/results/

# Exponer puerto de la API REST
EXPOSE 8000

# Healthcheck interno del contenedor
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de ejecución con Uvicorn
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
