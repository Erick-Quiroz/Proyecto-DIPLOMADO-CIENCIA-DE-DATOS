"""Esquemas Pydantic para validación de entrada y serialización de respuestas de la API."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Payload de entrada para solicitar una predicción de probabilidad de falla."""

    equipo: str = Field(..., description="Identificador o código del equipo industrial", example="EQ-195979")
    features: Dict[str, float] = Field(
        ...,
        description="Diccionario con las 44 variables predictoras exactas requeridas por el modelo",
    )
    model_override: Optional[str] = Field(
        None,
        description="Nombre de archivo de modelo específico opcional para evaluar (si no se especifica, usa el modelo activo)",
        example="modelo_random_forest.joblib",
    )


class PredictionResponse(BaseModel):
    """Respuesta estructurada de inferencia con probabilidad continua y nivel de riesgo."""

    equipo: str = Field(..., example="EQ-195979")
    probabilidad_falla_7_dias: float = Field(..., description="Probabilidad continua en escala decimal [0.0 - 1.0]", example=0.784)
    probabilidad_porcentaje: float = Field(..., description="Probabilidad expresada en porcentaje (0 - 100%)", example=78.4)
    nivel_riesgo: str = Field(..., description="Nivel de riesgo operativo (Bajo, Medio, Alto, Crítico)", example="Alto")
    modelo: str = Field(..., description="Nombre del algoritmo utilizado", example="Random Forest")
    version_modelo: str = Field(..., description="Nombre de archivo del modelo en producción", example="modelo_random_forest.joblib")
    prediccion_clase: int = Field(..., description="Clase binaria (0 = Operación normal, 1 = Alerta de falla)", example=1)
    recomendacion: str = Field(..., description="Acción de mantenimiento recomendada según el nivel de riesgo")
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class HealthResponse(BaseModel):
    """Respuesta de comprobación de salud del servicio."""

    status: str = Field("ok", example="ok")
    service: str = Field("predictive-maintenance-api", example="predictive-maintenance-api")
    active_model: str = Field(..., example="modelo_random_forest.joblib")
    version: str = Field("1.0.0", example="1.0.0")


class ModelInfo(BaseModel):
    """Metadatos de un modelo serializado disponible en el sistema."""

    name: str = Field(..., example="Random Forest")
    version: str = Field(..., example="v001")
    filename: str = Field(..., example="modelo_random_forest.joblib")
    active: bool = Field(..., example=True)
    family: str = Field(..., example="Ensemble (Bagging)")
    is_laboratory: bool = Field(..., example=False)


class ModelsListResponse(BaseModel):
    """Listado completo de modelos disponibles y modelo activo actual."""

    models: List[ModelInfo]
    active_model: str


class SetActiveModelRequest(BaseModel):
    """Solicitud para cambiar el modelo activo en producción."""

    model_filename: str = Field(..., example="modelo_random_forest.joblib")


class SetActiveModelResponse(BaseModel):
    """Confirmación del cambio de modelo activo."""

    status: str = "ok"
    message: str
    active_model: str
