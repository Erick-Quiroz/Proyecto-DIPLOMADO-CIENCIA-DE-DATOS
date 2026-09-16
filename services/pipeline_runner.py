"""Orquestador completo del pipeline de Ciencia de Datos de extremo a extremo."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import pandas as pd

# Asegurar ruta raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.data_cleaning.validator import DataValidator
from services.data_cleaning.config import CleaningConfig
from services.data_cleaning.pipeline import CleaningPipeline
from services.feature_engineering.config import FeatureConfig
from services.feature_engineering.pipeline import FeaturePipeline
from services.eda.config import EDAConfig
from services.eda.pipeline import EDAPipeline
from services.modeling.config import ModelingConfig
from services.modeling.pipeline import ModelingPipeline
from services.evaluation.config import EvaluationConfig
from services.evaluation.pipeline import EvaluationPipeline


class FullPipelineRunner:
    """Ejecuta secuencialmente todo el flujo de Ciencia de Datos desde los CSVs crudos."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT
        self.raw_data_dir = self.base_dir / "data" / "raw"
        self.processed_data_dir = self.base_dir / "data" / "processed"
        self.results_dir = self.base_dir / "results"
        self.models_dir = self.base_dir / "models"
        self.reports_dir = self.base_dir / "reports"

        self.validator = DataValidator(self.raw_data_dir)

    def run_pipeline(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de Ciencia de Datos en orden estricto:

        1. Validación de Datasets Crudos
        2. Limpieza y Auditoría de Datos
        3. Ingeniería de Características y Partición Cronológica
        4. Análisis Exploratorio de Datos (EDA)
        5. Modelado y Entrenamiento (Random Forest, Reg. Logística, XGBoost)
        6. Evaluación de Modelos y Generación de Métricas Finales
        """
        start_time = time.time()
        logs: list = []

        def log_step(step_idx: int, total_steps: int, msg: str):
            logs.append(f"[{time.strftime('%H:%M:%S')}] Paso {step_idx}/{total_steps}: {msg}")
            if progress_callback:
                progress_callback(step_idx, total_steps, msg)

        TOTAL_STEPS = 6

        # =========================================================================
        # PASO 1: VALIDACIÓN PREVIA DE ARCHIVOS CRUDOS
        # =========================================================================
        log_step(1, TOTAL_STEPS, "Validando estructura, esquemas y tipos de datos crudos...")
        val_summary = self.validator.validate_all_mandatory_datasets()

        if not val_summary["can_execute_pipeline"]:
            err_msg = "Error de validación: Existen datasets faltantes o con formato incorrecto."
            if val_summary["missing_datasets"]:
                err_msg += f" Faltan: {', '.join(val_summary['missing_datasets'])}."
            if val_summary["datasets_with_errors"]:
                err_msg += f" Errores en: {', '.join(val_summary['datasets_with_errors'])}."
            raise ValueError(err_msg)

        # =========================================================================
        # PASO 2: LIMPIEZA Y AUDITORÍA DE DATOS
        # =========================================================================
        log_step(2, TOTAL_STEPS, "Ejecutando limpieza, filtrado de horizonte y auditoría de atípicos...")
        cleaning_cfg = CleaningConfig(base_dir=self.base_dir)
        cleaning_pipeline = CleaningPipeline(cleaning_cfg)
        cleaning_result = cleaning_pipeline.run()

        df_validas = cleaning_result["df_validas"]
        stats_cleaning = cleaning_result["stats"]

        # =========================================================================
        # PASO 3: INGENIERÍA DE CARACTERÍSTICAS Y SPLIT CRONOLÓGICO
        # =========================================================================
        log_step(3, TOTAL_STEPS, "Generando variables históricas (7D/1D) y partición temporal...")
        feature_cfg = FeatureConfig(base_dir=self.base_dir)
        feature_pipeline = FeaturePipeline(feature_cfg)
        feature_result = feature_pipeline.run(df_validas, stats_cleaning=stats_cleaning)

        df_full = feature_result["df_full"]
        feature_columns = feature_result["feature_columns"]
        X_train = feature_result["X_train"]
        y_train = feature_result["y_train"]
        X_valid = feature_result["X_valid"]
        y_valid = feature_result["y_valid"]
        X_test = feature_result["X_test"]
        y_test = feature_result["y_test"]

        # =========================================================================
        # PASO 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
        # =========================================================================
        log_step(4, TOTAL_STEPS, "Ejecutando análisis exploratorio (EDA) y regenerando visualizaciones...")
        eda_cfg = EDAConfig(
            raw_data_dir=self.raw_data_dir,
            figures_dir=self.reports_dir / "figures" / "eda",
            tables_dir=self.reports_dir / "tables" / "eda",
        )
        eda_pipeline = EDAPipeline(eda_cfg)
        eda_result = eda_pipeline.run()

        # =========================================================================
        # PASO 5: MODELADO Y ENTRENAMIENTO DE MODELOS CANDIDATOS
        # =========================================================================
        log_step(5, TOTAL_STEPS, "Entrenando modelos candidatos (Random Forest, Regresión Logística, XGBoost)...")
        modeling_cfg = ModelingConfig(
            base_dir=self.base_dir,
            processed_data_dir=self.processed_data_dir,
            models_dir=self.models_dir,
            results_dir=self.results_dir / "modeling",
        )
        modeling_pipeline = ModelingPipeline(modeling_cfg)
        modeling_result = modeling_pipeline.run()

        # =========================================================================
        # PASO 6: EVALUACIÓN, SELECCIÓN Y CÁLCULO DE MÉTRICAS FINALES
        # =========================================================================
        log_step(6, TOTAL_STEPS, "Evaluando desempeño en conjunto de prueba independiente (X_test)...")
        evaluation_cfg = EvaluationConfig(
            base_dir=self.base_dir,
            processed_data_dir=self.processed_data_dir,
            models_dir=self.models_dir,
            results_dir=self.results_dir / "evaluation",
        )
        evaluation_pipeline = EvaluationPipeline(evaluation_cfg)
        evaluation_result = evaluation_pipeline.run()

        total_elapsed = round(time.time() - start_time, 2)
        log_step(6, TOTAL_STEPS, f"Pipeline completado exitosamente en {total_elapsed} segundos.")

        # Consolidar resumen de resultados
        selected_info = evaluation_result.get("selected_model_info", {})
        best_name = selected_info.get("display_name", selected_info.get("nombre", "Random Forest"))

        summary = {
            "status": "success",
            "elapsed_seconds": total_elapsed,
            "logs": logs,
            "datasets_validated": val_summary["datasets"],
            "filas_totales_raw": stats_cleaning.get("filas_iniciales", len(df_validas)),
            "filas_validas": len(df_validas),
            "equipos_unicos": int(df_validas["Identificador_Equipo"].nunique()),
            "total_features": len(feature_columns),
            "feature_columns": feature_columns,
            "train_size": len(X_train),
            "valid_size": len(X_valid),
            "test_size": len(X_test),
            "tasa_falla_train_pct": round(float(y_train.mean()) * 100, 2),
            "tasa_falla_test_pct": round(float(y_test.mean()) * 100, 2),
            "tabla_comparacion": evaluation_result.get("comparison_df"),
            "best_model_name": best_name,
            "eda_tables_count": eda_result.get("tables_generated", 0),
            "eda_figures_count": eda_result.get("figures_generated", 0),
        }

        return summary


def main():
    """Ejecución directa desde terminal."""
    runner = FullPipelineRunner()
    print("Iniciando ejecución completa del pipeline...")
    res = runner.run_pipeline(progress_callback=lambda s, t, m: print(f"[{s}/{t}] {m}"))
    print("\n" + "=" * 60)
    print("RESUMEN DE EJECUCIÓN DEL PIPELINE:")
    print(f" - Estado: {res['status']}")
    print(f" - Tiempo: {res['elapsed_seconds']} s")
    print(f" - Filas válidas: {res['filas_validas']:,d}")
    print(f" - Equipos únicos: {res['equipos_unicos']}")
    print(f" - Modelo ganador: {res['best_model_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
