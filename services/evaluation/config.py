"""Configuración de rutas, umbrales y artefactos para la Fase 7.5 de Evaluación."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class EvaluationConfig:
    """Configuración central para la evaluación, cálculo de métricas y exportación."""

    # Rutas base y directorios
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    processed_data_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    confusion_matrices_dir: Path = field(default=None)
    roc_curves_dir: Path = field(default=None)
    reports_figures_dir: Path = field(default=None)

    # Nombres de archivos de datos
    dataset_modelado_file: str = "dataset_modelado.csv"
    train_features_file: str = "X_train.csv"
    train_target_file: str = "y_train.csv"
    valid_features_file: str = "X_valid.csv"
    valid_target_file: str = "y_valid.csv"
    test_features_file: str = "X_test.csv"
    test_target_file: str = "y_test.csv"

    # Mapeo de modelos candidatos y nombres posibles de archivo
    models_to_evaluate: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "regresion_logistica": {
                "display_name": "Regresión Logística",
                "family": "Modelo Lineal Generalizado",
                "filenames": ["modelo_regresion_logistica.joblib", "regresion_logistica.pkl"],
            },
            "random_forest": {
                "display_name": "Random Forest",
                "family": "Ensemble (Bagging)",
                "filenames": ["modelo_random_forest.joblib", "random_forest.pkl"],
            },
            "xgboost": {
                "display_name": "XGBoost",
                "family": "Ensemble (Boosting)",
                "filenames": ["modelo_xgboost.joblib", "xgboost.pkl"],
            },
        }
    )

    # Nombres de archivos de salida requeridos
    comparison_table_file: str = "tabla_comparacion_modelos.csv"
    metrics_file: str = "metricas_modelos.csv"
    final_predictions_file: str = "predicciones_modelo_final.csv"
    all_predictions_test_file: str = "predicciones_todos_modelos_test.csv"
    stability_report_file: str = "diagnostico_estabilidad.csv"

    # Mapeo de clases
    class_labels: Dict[int, str] = field(
        default_factory=lambda: {
            0: "No se espera falla en los próximos 7 días",
            1: "Se espera falla en los próximos 7 días",
        }
    )

    def __post_init__(self):
        if self.processed_data_dir is None:
            self.processed_data_dir = self.base_dir / "data" / "processed"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results" / "evaluation"
        if self.confusion_matrices_dir is None:
            self.confusion_matrices_dir = self.results_dir / "matrices_confusion"
        if self.roc_curves_dir is None:
            self.roc_curves_dir = self.results_dir / "curvas_roc"
        if self.reports_figures_dir is None:
            self.reports_figures_dir = self.base_dir / "reports" / "figures"

        # Garantizar creación de todos los directorios destino
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
        self.roc_curves_dir.mkdir(parents=True, exist_ok=True)
        self.reports_figures_dir.mkdir(parents=True, exist_ok=True)
