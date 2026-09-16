#!/bin/sh
set -e

# Inicializar git local si no existe para que DVC reconozca el repositorio
if [ ! -d ".git" ]; then
    git init -q
fi

# Descargar datos procesados y modelos desde MinIO Cloud si no están en la imagen
if [ ! -f "data/processed/dataset_modelado.csv" ] || [ ! -f "models/modelo_random_forest.joblib" ]; then
    echo "Descargando datos y modelos desde MinIO mediante DVC..."
    dvc pull -f || true
fi

# Iniciar Dashboard Streamlit
exec streamlit run services/dashboard/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
