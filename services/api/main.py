"""Aplicación principal de FastAPI para el backend de Serving (CRISP-DM Fase 7.6)."""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configurar ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.api.config import APIConfig
from services.api.predictor import ModelPredictorManager
from services.api.routes import health_router, models_router, prediction_router

config = APIConfig(base_dir=BASE_DIR)
predictor_manager = ModelPredictorManager(config)

app = FastAPI(
    title=config.service_title,
    version=config.service_version,
    description=config.service_description,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS para permitir comunicación local fluida con Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(health_router)
app.include_router(models_router)
app.include_router(prediction_router)


@app.get("/", tags=["Root"], summary="Información del servicio")
def root_info():
    """Endpoint raíz con información de bienvenida y enlaces a documentación."""
    return {
        "message": "Bienvenido al API de Mantenimiento Predictivo - Área de Helados",
        "service": config.service_name,
        "version": config.service_version,
        "active_model": predictor_manager._active_model_name,
        "documentation": "/docs",
        "health_check": "/health",
        "models_endpoint": "/models",
        "prediction_endpoint": "/predict",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api.main:app", host="127.0.0.1", port=8502, reload=True)
