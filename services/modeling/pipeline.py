"""Pipeline de entrenamiento y generación de predicciones para la Fase 7.4 de Modelado."""

import time
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from services.modeling.config import ModelingConfig
from services.modeling.models import CandidateModelsFactory
from services.modeling.exporter import ModelingExporter


class ModelingPipeline:
    """Orquestador de la fase de modelado de clasificación binaria (CRISP-DM 7.4)."""

    def __init__(self, config: ModelingConfig = None):
        self.config = config or ModelingConfig()
        self.factory = CandidateModelsFactory(self.config)
        self.exporter = ModelingExporter(self.config)

    def load_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.DataFrame,
    ]:
        """Carga directamente los conjuntos particionados cronológicamente sin alteraciones."""
        data_dir = self.config.processed_data_dir

        x_train_path = data_dir / self.config.train_features_file
        y_train_path = data_dir / self.config.train_target_file
        x_valid_path = data_dir / self.config.valid_features_file
        y_valid_path = data_dir / self.config.valid_target_file
        x_test_path = data_dir / self.config.test_features_file
        y_test_path = data_dir / self.config.test_target_file
        dataset_path = data_dir / self.config.dataset_modelado_file

        for path in [x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path]:
            if not path.exists():
                raise FileNotFoundError(f"Archivo requerido no encontrado: {path}")

        X_train = pd.read_csv(x_train_path, encoding="utf-8-sig")
        y_train = pd.read_csv(y_train_path, encoding="utf-8-sig").squeeze("columns")
        X_valid = pd.read_csv(x_valid_path, encoding="utf-8-sig")
        y_valid = pd.read_csv(y_valid_path, encoding="utf-8-sig").squeeze("columns")
        X_test = pd.read_csv(x_test_path, encoding="utf-8-sig")
        y_test = pd.read_csv(y_test_path, encoding="utf-8-sig").squeeze("columns")

        # Asegurar tipo entero para el target
        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        y_test = y_test.astype(int)

        # Cargar metadatos del conjunto de validación si está disponible dataset_modelado.csv
        df_valid_meta = None
        if dataset_path.exists():
            df_full = pd.read_csv(dataset_path, encoding="utf-8-sig")
            if "Conjunto_Temporal" in df_full.columns:
                mask_val = df_full["Conjunto_Temporal"].isin(["Validacion", "Validación", "VALIDACION"])
                if mask_val.sum() == len(X_valid):
                    df_valid_meta = df_full[mask_val].reset_index(drop=True)

        return X_train, X_valid, X_test, y_train, y_valid, y_test, df_valid_meta

    def run(self) -> Dict[str, Any]:
        """Ejecuta el flujo completo de entrenamiento, inferencia en validación y persistencia."""
        print("[Modeling Pipeline] 1. Cargando particiones cronológicas...")
        X_train, X_valid, X_test, y_train, y_valid, y_test, df_valid_meta = self.load_data()

        print(f" - X_train: {X_train.shape} | y_train: {y_train.shape} (Tasa fallas: {y_train.mean():.2%})")
        print(f" - X_valid: {X_valid.shape} | y_valid: {y_valid.shape} (Tasa fallas: {y_valid.mean():.2%})")
        print(f" - X_test : {X_test.shape} | y_test : {X_test.shape} (Tasa fallas: {y_test.mean():.2%}) [Manteniendo intacto]")

        models_catalog = self.factory.build_all()
        trained_models = {}
        predictions_valid = {}
        summary_rows = []
        saved_model_paths = {}

        print("\n[Modeling Pipeline] 2. Entrenando modelos candidatos sobre X_train...")
        for key, model_info in models_catalog.items():
            display_name = model_info["display_name"]
            estimator = model_info["estimator"]
            file_name = model_info["file_name"]

            print(f" -> Entrenando {display_name}...")
            start_time = time.time()
            estimator.fit(X_train, y_train)
            train_duration = time.time() - start_time

            # Inferencia sobre conjunto de validación
            y_pred_valid = estimator.predict(X_valid)
            y_proba_valid = estimator.predict_proba(X_valid)[:, 1]

            # Inferencia sobre conjunto de entrenamiento (para trazabilidad/diagnóstico inicial)
            y_pred_train = estimator.predict(X_train)
            y_proba_train = estimator.predict_proba(X_train)[:, 1]

            # Guardar modelo en disco
            saved_path = self.exporter.save_model(estimator, file_name)
            saved_model_paths[key] = saved_path

            trained_models[key] = {
                "estimator": estimator,
                "display_name": display_name,
                "saved_path": saved_path,
                "train_duration_sec": train_duration,
            }

            predictions_valid[key] = {
                "display_name": display_name,
                "valid_pred_file": model_info["valid_pred_file"],
                "y_pred": y_pred_valid,
                "y_proba": y_proba_valid,
            }

            # Construir fila de resumen
            summary_rows.append(
                {
                    "id_modelo": key,
                    "nombre_modelo": display_name,
                    "familia_algoritmo": model_info["family"],
                    "archivo_modelo": file_name,
                    "escalamiento_interno_pipeline": model_info["has_internal_scaler"],
                    "n_muestras_entrenamiento": len(X_train),
                    "n_variables_predictoras": X_train.shape[1],
                    "tasa_positiva_train_pct": round(float(y_train.mean() * 100), 2),
                    "hiperparametros": str(model_info["params"]),
                    "tiempo_entrenamiento_seg": round(train_duration, 4),
                    "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            print(f"    Completado en {train_duration:.3f}s. Guardado en {saved_path}")

        print("\n[Modeling Pipeline] 3. Exportando tablas de resultados y resumen...")
        valid_paths = self.exporter.save_validation_predictions(
            y_valid, predictions_valid, df_valid_meta=df_valid_meta
        )
        summary_path = self.exporter.save_summary_table(summary_rows)

        return {
            "trained_models": trained_models,
            "predictions_valid": predictions_valid,
            "summary_rows": summary_rows,
            "summary_path": summary_path,
            "saved_model_paths": saved_model_paths,
            "valid_paths": valid_paths,
            "shapes": {
                "X_train": X_train.shape,
                "X_valid": X_valid.shape,
                "X_test": X_test.shape,
                "y_train_fallas": int(y_train.sum()),
                "y_valid_fallas": int(y_valid.sum()),
                "y_test_fallas": int(y_test.sum()),
            },
        }
