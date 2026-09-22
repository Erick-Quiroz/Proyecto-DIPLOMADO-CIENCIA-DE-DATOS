"""Rutas para consultar y gestionar los modelos de Machine Learning disponibles y activos."""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from services.api.schemas import (
    ModelsListResponse,
    SetActiveModelRequest,
    SetActiveModelResponse,
    ModelInfo,
)
from services.api.predictor import ModelPredictorManager

router = APIRouter(prefix="/models", tags=["Models"])


def get_manager() -> ModelPredictorManager:
    from services.api.main import predictor_manager
    return predictor_manager


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="Listar todos los modelos disponibles",
    description="Devuelve la lista de modelos entrenados (Fase 7.4 y Laboratorio) e indica cuál está activo.",
)
def list_models(manager: ModelPredictorManager = Depends(get_manager)) -> ModelsListResponse:
    return manager.list_models()


@router.get(
    "/active",
    response_model=ModelInfo,
    summary="Consultar el modelo activo actual",
)
def get_active_model(manager: ModelPredictorManager = Depends(get_manager)) -> ModelInfo:
    models_resp = manager.list_models()
    active_name = models_resp.active_model
    for m in models_resp.models:
        if m.filename == active_name:
            return m
    raise HTTPException(status_code=404, detail="Modelo activo no encontrado en la lista.")


@router.post(
    "/set-active",
    response_model=SetActiveModelResponse,
    summary="Establecer un modelo como activo en producción",
    description="Permite promover un modelo (incluyendo modelos de laboratorio) para su consumo en /predict.",
)
def set_active_model(
    request: SetActiveModelRequest,
    manager: ModelPredictorManager = Depends(get_manager),
) -> SetActiveModelResponse:
    try:
        new_active = manager.set_active_model(request.model_filename)
        return SetActiveModelResponse(
            status="ok",
            message=f"El modelo '{new_active}' ha sido establecido exitosamente como activo en producción.",
            active_model=new_active,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/metrics",
    summary="Consultar métricas de evaluación de modelos",
    description="Devuelve las métricas de rendimiento en particiones de Validación (X_valid) y Prueba (X_test).",
)
def get_model_metrics() -> Dict[str, Any]:
    import pandas as pd
    from services.api.config import APIConfig

    cfg = APIConfig()
    eval_dir = cfg.base_dir / "results" / "evaluation"
    stability_path = eval_dir / "diagnostico_estabilidad.csv"
    comp_path = eval_dir / "tabla_comparacion_modelos.csv"

    res: Dict[str, Any] = {
        "status": "ok",
        "validacion": [],
        "prueba": [],
    }

    if stability_path.exists():
        df_stab = pd.read_csv(stability_path, encoding="utf-8-sig")
        val_mask = df_stab["particion"].isin(["Valid", "Validacion", "Validación"])
        test_mask = df_stab["particion"].isin(["Test", "Prueba", "PRUEBA"])
        res["validacion"] = df_stab[val_mask].to_dict(orient="records")
        res["prueba"] = df_stab[test_mask].to_dict(orient="records")
    elif comp_path.exists():
        df_comp = pd.read_csv(comp_path, encoding="utf-8-sig")
        res["prueba"] = df_comp.to_dict(orient="records")

    return res

