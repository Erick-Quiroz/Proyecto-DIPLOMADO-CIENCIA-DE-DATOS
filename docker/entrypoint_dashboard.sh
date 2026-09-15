#!/bin/sh
set -e

# Descargar datos procesados y modelos desde MinIO si no están en la imagen
if [ ! -f "data/processed/dataset_modelado.csv" ] || [ ! -f "models/modelo_random_forest.joblib" ]; then
    echo "Descargando datos y modelos desde MinIO mediante DVC..."
    dvc pull || true
fi

# Iniciar Dashboard Streamlit
exec streamlit run services/dashboard/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
