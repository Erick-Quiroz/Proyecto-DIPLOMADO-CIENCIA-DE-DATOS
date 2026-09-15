# =====================================================================
# DOCKERFILE: Streamlit Frontend Dashboard (CRISP-DM Fase 7.6)
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

# Copiar código del dashboard, datos procesados, modelos y resultados
COPY services/ /app/services/
COPY data/processed/ /app/data/processed/
COPY models/ /app/models/
COPY results/ /app/results/

# Exponer puerto de Streamlit
EXPOSE 8501

# Healthcheck interno del contenedor
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de ejecución con Streamlit
CMD ["streamlit", "run", "services/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
