"""Pipeline integral de evaluación, comparación y selección de modelos (CRISP-DM 7.5)."""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd

from services.evaluation.config import EvaluationConfig
from services.evaluation.evaluator import ModelEvaluator
from services.evaluation.selector import ModelSelector
from services.evaluation.exporter import EvaluationExporter


class EvaluationPipeline:
    """Orquestador de la fase de Evaluación y Resultados (CRISP-DM 7.5)."""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.evaluator = ModelEvaluator(self.config)
        self.selector = ModelSelector()
        self.exporter = EvaluationExporter(self.config)

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
        """Carga los conjuntos de datos particionados y los metadatos de prueba."""
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
        y_train = pd.read_csv(y_train_path, encoding="utf-8-sig").squeeze("columns").astype(int)
        X_valid = pd.read_csv(x_valid_path, encoding="utf-8-sig")
        y_valid = pd.read_csv(y_valid_path, encoding="utf-8-sig").squeeze("columns").astype(int)
        X_test = pd.read_csv(x_test_path, encoding="utf-8-sig")
        y_test = pd.read_csv(y_test_path, encoding="utf-8-sig").squeeze("columns").astype(int)

        # Cargar metadatos del conjunto de prueba
        df_test_meta = None
        if dataset_path.exists():
            df_full = pd.read_csv(dataset_path, encoding="utf-8-sig")
            if "Conjunto_Temporal" in df_full.columns:
                mask_test = df_full["Conjunto_Temporal"].isin(["Prueba", "Test", "TEST", "PRUEBA"])
                if mask_test.sum() == len(X_test):
                    df_test_meta = df_full[mask_test].reset_index(drop=True)

        return X_train, X_valid, X_test, y_train, y_valid, y_test, df_test_meta

    def load_models(self) -> Dict[str, Dict[str, Any]]:
        """Carga los modelos entrenados desde models/ soportando .joblib y .pkl."""
        loaded_models = {}

        for key, info in self.config.models_to_evaluate.items():
            loaded = False
            for filename in info["filenames"]:
                model_path = self.config.models_dir / filename
                if model_path.exists():
                    estimator = joblib.load(model_path)
                    loaded_models[key] = {
                        "display_name": info["display_name"],
                        "family": info["family"],
                        "estimator": estimator,
                        "file_path": model_path,
                    }
                    loaded = True
                    break

            if not loaded:
                raise FileNotFoundError(
                    f"No se encontró el modelo para '{key}'. Buscado en {info['filenames']} dentro de {self.config.models_dir}"
                )

        return loaded_models

    def run(self) -> Dict[str, Any]:
        """Ejecuta la evaluación completa, comparación y persistencia de artefactos."""
        X_train, X_valid, X_test, y_train, y_valid, y_test, df_test_meta = self.load_data()
        loaded_models = self.load_models()

        # 1. Evaluación sobre conjunto de prueba X_test
        evaluation_results = {}
        for key, model_data in loaded_models.items():
            metrics = self.evaluator.evaluate_model_on_split(
                model_data["estimator"], X_test, y_test
            )
            evaluation_results[key] = {
                "display_name": model_data["display_name"],
                "family": model_data["family"],
                "estimator": model_data["estimator"],
                "metrics": metrics,
            }

        # 2. Diagnóstico de estabilidad y detección de overfitting (Train vs Valid vs Test)
        stability_rows = self.evaluator.generate_stability_analysis(
            loaded_models, X_train, y_train, X_valid, y_valid, X_test, y_test
        )

        # 3. Gráficos de matrices de confusión y curvas ROC
        cm_plots = self.evaluator.plot_confusion_matrices(evaluation_results)
        roc_plot = self.evaluator.plot_comparative_roc_curves(y_test, evaluation_results)

        # 4. Tabla comparativa y selección del modelo ganador
        comparison_df = self.selector.build_comparison_table(evaluation_results)
        selected_model_info, justification = self.selector.select_best_model(comparison_df)

        # 5. Exportación de tablas y predicciones
        table_paths = self.exporter.save_comparison_and_metrics_tables(
            comparison_df, stability_rows
        )
        selected_estimator = loaded_models[selected_model_info["id_modelo"]]["estimator"]
        final_preds_path = self.exporter.save_final_predictions(
            selected_model_info, selected_estimator, X_test, y_test, df_test_meta=df_test_meta
        )
        all_preds_path = self.exporter.save_all_test_predictions(
            evaluation_results, y_test, df_test_meta=df_test_meta
        )

        return {
            "evaluation_results": evaluation_results,
            "comparison_df": comparison_df,
            "selected_model_info": selected_model_info,
            "justification": justification,
            "stability_rows": stability_rows,
            "cm_plots": cm_plots,
            "roc_plot": roc_plot,
            "table_paths": table_paths,
            "final_preds_path": final_preds_path,
            "all_preds_path": all_preds_path,
            "shapes": {
                "n_test": len(X_test),
                "n_features": X_test.shape[1],
                "n_fallas_test": int(y_test.sum()),
                "target_name": "Falla_En_Los_Siguientes_Siete_Dias",
            },
        }
