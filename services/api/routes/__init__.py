"""Rutas de endpoints de la API FastAPI."""

from services.api.routes.health import router as health_router
from services.api.routes.models import router as models_router
from services.api.routes.prediction import router as prediction_router

__all__ = ["health_router", "models_router", "prediction_router"]
