"""Gestor de inferencia, validación de variables y carga en memoria de modelos activos."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from services.api.config import APIConfig
from services.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    ModelsListResponse,
)


class ModelPredictorManager:
    """Gestiona la carga de modelos, validación estricta de características e inferencia."""

    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self._models_cache: Dict[str, Any] = {}
        self._active_model_name: str = self.config.get_active_model_filename()
        self._feature_columns: List[str] = self._load_expected_features()

        # Precargar modelo activo
        self.ensure_active_model_loaded()

    def _load_expected_features(self) -> List[str]:
        """Carga la lista canónica de 44 variables predictoras desde X_train."""
        x_train_path = self.config.base_dir / "data" / "processed" / "X_train.csv"
        if x_train_path.exists():
            df_head = pd.read_csv(x_train_path, nrows=1, encoding="utf-8-sig")
            return list(df_head.columns)
        # Lista canónica de respaldo en caso de emergencia
        return [
            "Temperatura_Proceso", "Vibracion_Equipo", "Presion_Sistema", "Corriente_Motor",
            "Caudal_Proceso", "Nivel_Sistema", "Velocidad_Accionamiento", "Horas_Operacion",
            "Temperatura_Proceso_Media_7D", "Temperatura_Proceso_Std_7D", "Temperatura_Proceso_Delta_1D",
            "Vibracion_Equipo_Media_7D", "Vibracion_Equipo_Std_7D", "Vibracion_Equipo_Delta_1D",
            "Presion_Sistema_Media_7D", "Presion_Sistema_Std_7D", "Presion_Sistema_Delta_1D",
            "Corriente_Motor_Media_7D", "Corriente_Motor_Std_7D", "Corriente_Motor_Delta_1D",
            "Carga_Electromecanica", "Ratio_Presion_Caudal", "Criticidad_Ordinal",
            "Tipo_Equipo_Bomba de circulación", "Tipo_Equipo_Compresor de refrigeración",
            "Tipo_Equipo_Congelador industrial", "Tipo_Equipo_Cámara frigorífica",
            "Tipo_Equipo_Dosificadora de helado", "Tipo_Equipo_Envasadora automática",
            "Tipo_Equipo_Homogeneizador", "Tipo_Equipo_Mezclador industrial",
            "Tipo_Equipo_Pasteurizador", "Tipo_Equipo_Selladora de envases",
            "Tipo_Equipo_Transportador de producto", "Tipo_Equipo_Túnel de congelación",
            "Proceso_Almacenamiento", "Proceso_Congelación", "Proceso_Envasado",
            "Proceso_Preparación", "Proceso_Refrigeración", "Línea_Línea de Helados 01",
            "Línea_Línea de Helados 02", "Línea_Línea de Helados 03", "Línea_Línea de Helados 04"
        ]

    def _find_model_file(self, filename: str) -> Optional[Path]:
        """Busca el archivo del modelo en models/ o models/laboratorio/."""
        path_main = self.config.models_dir / filename
        if path_main.exists():
            return path_main

        path_lab = self.config.lab_models_dir / filename
        if path_lab.exists():
            return path_lab

        return None

    def get_model(self, filename: str) -> Any:
        """Obtiene un modelo desde la memoria o lo carga desde disco con validación."""
        if filename not in self._models_cache:
            file_path = self._find_model_file(filename)
            if not file_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Modelo '{filename}' no encontrado en el directorio de modelos.",
                )
            try:
                self._models_cache[filename] = joblib.load(file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error al deserializar el modelo '{filename}': {str(e)}",
                )
        return self._models_cache[filename]

    def ensure_active_model_loaded(self) -> None:
        """Garantiza que el modelo activo esté cargado en memoria."""
        self._active_model_name = self.config.get_active_model_filename()
        try:
            self.get_model(self._active_model_name)
        except Exception:
            # Si falla, buscar cualquier modelo disponible como fallback
            models = self.list_models()
            if models.models:
                self._active_model_name = models.models[0].filename
                self.config.set_active_model_filename(self._active_model_name)
                self.get_model(self._active_model_name)

    def set_active_model(self, filename: str) -> str:
        """Actualiza y persiste el modelo activo en producción."""
        # Verificar que el modelo cargue correctamente
        self.get_model(filename)
        self._active_model_name = filename
        self.config.set_active_model_filename(filename)
        return self._active_model_name

    def list_models(self) -> ModelsListResponse:
        """Escanea todos los modelos serializados en models/ y models/laboratorio/."""
        models_list: List[ModelInfo] = []
        active_curr = self.config.get_active_model_filename()

        search_dirs = [self.config.models_dir, self.config.lab_models_dir]
        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            for file_path in sorted(list(s_dir.glob("*.joblib")) + list(s_dir.glob("*.pkl"))):
                stem = file_path.stem
                fname = file_path.name
                is_lab = "laboratorio" in str(file_path).lower()

                if "random_forest" in stem:
                    name = "Random Forest"
                    family = "Ensemble (Bagging)"
                elif "xgboost" in stem:
                    name = "XGBoost"
                    family = "Ensemble (Boosting)"
                elif "regresion_logistica" in stem:
                    name = "Regresión Logística"
                    family = "Modelo Lineal Generalizado"
                else:
                    name = stem.replace("_", " ").title()
                    family = "Modelo Experimental"

                version = stem.split("_")[-1] if "_" in stem else "v001"
                models_list.append(
                    ModelInfo(
                        name=name,
                        version=version,
                        filename=fname,
                        active=(fname == active_curr),
                        family=family,
                        is_laboratory=is_lab,
                    )
                )

        return ModelsListResponse(models=models_list, active_model=active_curr)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Valida las variables de entrada y ejecuta la inferencia predictiva."""
        # Determinar modelo a usar
        model_filename = request.model_override or self._active_model_name
        model = self.get_model(model_filename)

        # Validar variables predictoras
        missing_features = [col for col in self._feature_columns if col not in request.features]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Faltan {len(missing_features)} variables predictoras requeridas: {missing_features[:5]}...",
            )

        # Construir vector de características en el orden exacto
        ordered_values = [float(request.features[col]) for col in self._feature_columns]
        X_df = pd.DataFrame([ordered_values], columns=self._feature_columns)

        # Validar valores NaN o nulos
        if X_df.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="El vector de variables contiene valores nulos o inválidos.",
            )

        try:
            # Inferencia de probabilidad
            proba_arr = model.predict_proba(X_df)
            y_proba = float(proba_arr[0, 1])
            y_pred = int(y_proba >= 0.5)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error durante el cálculo de predict_proba(): {str(e)}",
            )

        # Calcular nivel de riesgo y recomendación
        risk_info = self.config.get_risk_level(y_proba)

        # Identificar nombre legible del modelo
        if "random_forest" in model_filename:
            model_name = "Random Forest"
        elif "xgboost" in model_filename:
            model_name = "XGBoost"
        elif "regresion_logistica" in model_filename:
            model_name = "Regresión Logística"
        else:
            model_name = model_filename.replace(".joblib", "").replace(".pkl", "").title()

        return PredictionResponse(
            equipo=request.equipo,
            probabilidad_falla_7_dias=round(y_proba, 4),
            probabilidad_porcentaje=round(y_proba * 100, 2),
            nivel_riesgo=risk_info["label"],
            modelo=model_name,
            version_modelo=model_filename,
            prediccion_clase=y_pred,
            recomendacion=risk_info["action"],
        )
