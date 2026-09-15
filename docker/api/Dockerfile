# =====================================================================
# DOCKERFILE: FastAPI Backend Inferencia (CRISP-DM Fase 7.6)
# Mantenimiento Predictivo - Área de Helados
# =====================================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Instalar dependencias del sistema (curl para healthcheck, git para DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Crear estructura de carpetas
RUN mkdir -p /app/data/processed /app/data/raw /app/models /app/results

# Copiar configuraciones de DVC
COPY .dvc/ /app/.dvc/
COPY .dvcignore /app/.dvcignore

# Copiar código fuente y carpetas de la aplicación
COPY services/ /app/services/
COPY models/ /app/models/
COPY data/ /app/data/
COPY results/ /app/results/

# Copiar y configurar script de arranque
COPY docker/entrypoint_api.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
