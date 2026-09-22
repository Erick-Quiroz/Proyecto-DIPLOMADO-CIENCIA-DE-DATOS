"""Configuración de rutas, hiperparámetros y artefactos para la Fase 7.4 de Modelado."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class ModelingConfig:
    """Configuración central para el entrenamiento y serialización de modelos candidatos."""

    # Rutas base y directorios
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    processed_data_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    results_dir: Path = field(default=None)

    # Nombres de archivos de entrada (Fase 7.3)
    dataset_modelado_file: str = "dataset_modelado.csv"
    train_features_file: str = "X_train.csv"
    train_target_file: str = "y_train.csv"
    valid_features_file: str = "X_valid.csv"
    valid_target_file: str = "y_valid.csv"
    test_features_file: str = "X_test.csv"
    test_target_file: str = "y_test.csv"

    # Nombres de archivos de salida para modelos (models/)
    logistic_regression_model_file: str = "modelo_regresion_logistica.joblib"
    random_forest_model_file: str = "modelo_random_forest.joblib"
    xgboost_model_file: str = "modelo_xgboost.joblib"

    # Nombres de archivos de salida de resultados (results/modeling/)
    summary_file: str = "resumen_modelos.csv"
    consolidated_valid_predictions_file: str = "predicciones_probabilidades_validacion.csv"
    detailed_valid_predictions_file: str = "predicciones_equipos_validacion.csv"
    lr_valid_predictions_file: str = "predicciones_val_regresion_logistica.csv"
    rf_valid_predictions_file: str = "predicciones_val_random_forest.csv"
    xgb_valid_predictions_file: str = "predicciones_val_xgboost.csv"

    # Hiperparámetros de Regresión Logística
    logistic_regression_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42,
        }
    )

    # Hiperparámetros de Random Forest
    random_forest_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    # Hiperparámetros de XGBoost
    xgboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 8.38,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    def __post_init__(self):
        if self.processed_data_dir is None:
            self.processed_data_dir = self.base_dir / "data" / "processed"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results" / "modeling"

        # Garantizar creación de directorios destino
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
