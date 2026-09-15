"""Exportador de tablas de comparación, métricas y predicciones finales de evaluación."""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from services.evaluation.config import EvaluationConfig


class EvaluationExporter:
    """Gestiona el almacenamiento de todas las tablas de reporte y predicciones de la Fase 7.5."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def save_comparison_and_metrics_tables(
        self,
        comparison_df: pd.DataFrame,
        stability_rows: List[Dict[str, Any]],
    ) -> Dict[str, Path]:
        """Guarda tabla_comparacion_modelos.csv, metricas_modelos.csv y diagnostico_estabilidad.csv."""
        saved_paths = {}

        # 1. Tabla comparativa limpia (formato estándar requerido)
        clean_comparison = comparison_df[
            ["Modelo", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        ].copy()
        comp_path = self.config.results_dir / self.config.comparison_table_file
        clean_comparison.to_csv(comp_path, index=False, encoding="utf-8-sig")
        saved_paths["tabla_comparacion"] = comp_path

        # 2. Métricas completas y detalladas
        metrics_path = self.config.results_dir / self.config.metrics_file
        comparison_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        saved_paths["metricas_modelos"] = metrics_path

        # 3. Diagnóstico de estabilidad (Train / Valid / Test)
        df_stability = pd.DataFrame(stability_rows)
        stability_path = self.config.results_dir / self.config.stability_report_file
        df_stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
        saved_paths["diagnostico_estabilidad"] = stability_path

        return saved_paths

    def save_final_predictions(
        self,
        selected_model_info: Dict[str, Any],
        selected_estimator: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test_meta: pd.DataFrame = None,
    ) -> Path:
        """Genera el archivo predicciones_modelo_final.csv con identificación de equipos y probabilidades."""
        y_pred = selected_estimator.predict(X_test)
        y_proba = selected_estimator.predict_proba(X_test)[:, 1]

        # Construir base de datos
        if df_test_meta is not None:
            meta_cols = [
                col
                for col in [
                    "Identificador_Equipo",
                    "Nombre_Equipo",
                    "Tipo_Equipo",
                    "Línea",
                    "Proceso",
                    "Criticidad",
                    "Fecha_Observacion",
                ]
                if col in df_test_meta.columns
            ]
            df_out = df_test_meta[meta_cols].copy().reset_index(drop=True)
        else:
            df_out = pd.DataFrame(index=X_test.index)
            df_out["Identificador_Equipo"] = [f"EQ-{i+1:04d}" for i in range(len(X_test))]
            df_out["Fecha_Observacion"] = "2026-08-20"

        df_out["Falla_Real"] = y_test.values
        df_out["Prediccion"] = y_pred
        df_out["Probabilidad_Falla_7_Dias"] = y_proba.round(4)
        df_out["Probabilidad_Falla_7_Dias_Pct"] = (y_proba * 100).round(2)
        df_out["Modelo"] = selected_model_info["nombre_modelo"]

        out_path = self.config.results_dir / self.config.final_predictions_file
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    def save_all_test_predictions(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        y_test: pd.Series,
        df_test_meta: pd.DataFrame = None,
    ) -> Path:
        """Genera el archivo consolidado de predicciones de prueba para todos los modelos."""
        if df_test_meta is not None:
            meta_cols = [
                col
                for col in [
                    "Identificador_Equipo",
                    "Nombre_Equipo",
                    "Tipo_Equipo",
                    "Línea",
                    "Proceso",
                    "Criticidad",
                    "Fecha_Observacion",
                ]
                if col in df_test_meta.columns
            ]
            df_out = df_test_meta[meta_cols].copy().reset_index(drop=True)
        else:
            df_out = pd.DataFrame(index=y_test.index)

        df_out["Falla_Real"] = y_test.values

        for key, res in evaluation_results.items():
            name = res["display_name"]
            y_pred = res["metrics"]["y_pred"]
            y_proba = res["metrics"]["y_proba"]

            df_out[f"Prediccion_{name}"] = y_pred
            df_out[f"Probabilidad_Falla_{name}"] = y_proba.round(4)
            df_out[f"Probabilidad_Falla_{name} (%)"] = (y_proba * 100).round(2)

        out_path = self.config.results_dir / self.config.all_predictions_test_file
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
