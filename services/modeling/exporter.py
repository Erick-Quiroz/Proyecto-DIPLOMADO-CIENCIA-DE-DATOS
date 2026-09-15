"""Exportador y persistencia de modelos entrenados y artefactos de validación."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd

from services.modeling.config import ModelingConfig


class ModelingExporter:
    """Gestiona el almacenamiento seguro y trazable de modelos y tablas de resultados."""

    def __init__(self, config: ModelingConfig):
        self.config = config

    def save_model(self, estimator: Any, filename: str) -> Path:
        """Serializa y guarda el estimador o pipeline entrenado en models/."""
        output_path = self.config.models_dir / filename
        joblib.dump(estimator, output_path)
        return output_path

    def save_validation_predictions(
        self,
        y_valid: pd.Series,
        predictions_dict: Dict[str, Dict[str, Any]],
        df_valid_meta: pd.DataFrame = None,
    ) -> Dict[str, Path]:
        """Guarda los archivos individuales, el consolidado y el detalle por equipo con % de falla."""
        saved_paths = {}

        # 1. Crear DataFrame consolidado técnico
        consolidated_df = pd.DataFrame(index=y_valid.index)
        consolidated_df["y_real"] = y_valid.values

        for key, pred_info in predictions_dict.items():
            col_pred = f"pred_{key}"
            col_proba = f"proba_{key}"

            consolidated_df[col_pred] = pred_info["y_pred"]
            consolidated_df[col_proba] = pred_info["y_proba"]

            # Guardar archivo individual
            indiv_df = pd.DataFrame(
                {
                    "y_real": y_valid.values,
                    "prediccion_clase": pred_info["y_pred"],
                    "probabilidad_falla_7d": pred_info["y_proba"],
                },
                index=y_valid.index,
            )
            indiv_path = self.config.results_dir / pred_info["valid_pred_file"]
            indiv_df.to_csv(indiv_path, index=False, encoding="utf-8-sig")
            saved_paths[key] = indiv_path

        # Guardar consolidado técnico
        consolidated_path = (
            self.config.results_dir / self.config.consolidated_valid_predictions_file
        )
        consolidated_df.to_csv(consolidated_path, index=False, encoding="utf-8-sig")
        saved_paths["consolidated"] = consolidated_path

        # 2. Si se proveen metadatos de equipos, generar tabla detallada de negocio
        if df_valid_meta is not None:
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
                if col in df_valid_meta.columns
            ]
            detailed_df = df_valid_meta[meta_cols].copy().reset_index(drop=True)
            detailed_df["Falla_Real_7D"] = y_valid.values

            # Agregar probabilidades formateadas en porcentaje (%) y predicciones de clase
            for key, pred_info in predictions_dict.items():
                display = pred_info["display_name"]
                proba_arr = pred_info["y_proba"]
                pred_arr = pred_info["y_pred"]

                detailed_df[f"Prob_Falla_{display} (%)"] = (proba_arr * 100).round(2)
                detailed_df[f"Prediccion_{display}"] = pred_arr

            detailed_path = (
                self.config.results_dir / self.config.detailed_valid_predictions_file
            )
            detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
            saved_paths["detailed_equipos"] = detailed_path

        return saved_paths

    def save_summary_table(self, summary_rows: List[Dict[str, Any]]) -> Path:
        """Genera y guarda el archivo resumen_modelos.csv con los metadatos de los modelos."""
        df_summary = pd.DataFrame(summary_rows)
        summary_path = self.config.results_dir / self.config.summary_file
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_path
