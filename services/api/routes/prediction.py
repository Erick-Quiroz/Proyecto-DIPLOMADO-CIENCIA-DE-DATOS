"""Ruta principal de inferencia predictiva para estimar probabilidad de falla a 7 días."""

from fastapi import APIRouter, Depends, HTTPException
from services.api.schemas import PredictionRequest, PredictionResponse
from services.api.predictor import ModelPredictorManager

router = APIRouter(tags=["Prediction"])


def get_manager() -> ModelPredictorManager:
    from services.api.main import predictor_manager
    return predictor_manager


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Estimar probabilidad de falla en 7 días",
    description="Recibe el identificador del equipo y las 44 variables predictoras, ejecutando el modelo activo para calcular la probabilidad y nivel de riesgo.",
)
def predict_failure_probability(
    request: PredictionRequest,
    manager: ModelPredictorManager = Depends(get_manager),
) -> PredictionResponse:
    try:
        return manager.predict(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado durante la inferencia predictiva: {str(e)}",
        )
