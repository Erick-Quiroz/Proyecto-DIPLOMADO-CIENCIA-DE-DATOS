#!/bin/sh
set -e

# Inicializar git local si no existe para que DVC reconozca el repositorio
if [ ! -d ".git" ]; then
    git init -q
fi

# Descargar artefacto del modelo desde MinIO si no está en la imagen
if [ ! -f "models/modelo_random_forest.joblib" ]; then
    echo "Descargando modelo activo desde MinIO mediante DVC..."
    dvc pull -f models/modelo_random_forest.joblib.dvc || true
fi

# Iniciar servidor FastAPI en el puerto 8502
exec uvicorn services.api.main:app --host 0.0.0.0 --port 8502
