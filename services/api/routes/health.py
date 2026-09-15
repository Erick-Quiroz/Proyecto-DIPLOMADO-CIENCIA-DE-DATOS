"""Ruta del endpoint de comprobación de salud del servicio (Health Check)."""

from fastapi import APIRouter, Depends
from services.api.config import APIConfig
from services.api.schemas import HealthResponse
from services.api.predictor import ModelPredictorManager

router = APIRouter(tags=["Health"])


def get_manager() -> ModelPredictorManager:
    # Inyección de dependencia simple
    from services.api.main import predictor_manager
    return predictor_manager


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar estado de salud del servicio",
    description="Comprueba que la API esté en línea y devuelve el modelo activo cargado en producción.",
)
def health_check(manager: ModelPredictorManager = Depends(get_manager)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=manager.config.service_name,
        active_model=manager._active_model_name,
        version=manager.config.service_version,
    )
