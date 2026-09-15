#!/bin/sh
set -e

# Descargar artefacto del modelo desde MinIO si no está en la imagen
if [ ! -f "models/modelo_random_forest.joblib" ]; then
    echo "Descargando modelo activo desde MinIO mediante DVC..."
    dvc pull models/modelo_random_forest.joblib.dvc || true
fi

# Iniciar servidor FastAPI
exec uvicorn services.api.main:app --host 0.0.0.0 --port 8000
