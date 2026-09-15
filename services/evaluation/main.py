"""Punto de entrada CLI para la Fase 7.5: Evaluación y Resultados (CRISP-DM)."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Permitir ejecución directa o como módulo
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.evaluation.config import EvaluationConfig
from services.evaluation.pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Servicio de Evaluación, Comparación y Selección de Modelos (CRISP-DM 7.5)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directorio de datos procesados",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directorio de modelos entrenados",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directorio de salida para resultados de evaluación",
    )
    args = parser.parse_args()

    config = EvaluationConfig()
    if args.data_dir:
        config.processed_data_dir = Path(args.data_dir)
    if args.models_dir:
        config.models_dir = Path(args.models_dir)
    if args.results_dir:
        config.results_dir = Path(args.results_dir)

    pipeline = EvaluationPipeline(config)
    results = pipeline.run()

    shapes = results["shapes"]
    eval_res = results["evaluation_results"]
    comp_df = results["comparison_df"]
    selected_info = results["selected_model_info"]
    justification = results["justification"]

    # Cargar muestra de predicciones para ejemplos reales
    df_preds = pd.read_csv(results["final_preds_path"], encoding="utf-8-sig")

    print("\n" + "=" * 50)
    print("FASE 7.5 - EVALUACIÓN Y RESULTADOS")
    print("=" * 50)

    print("\nDatos de evaluación:")
    print(f"- X_test: {shapes['n_test']:,d} registros")
    print(f"- Variables predictoras: {shapes['n_features']}")
    print(f"- Variable objetivo: {shapes['target_name']} ({shapes['n_fallas_test']} fallas en test)")

    print("\n" + "-" * 50)
    print("EVALUACIÓN DE MODELOS")
    print("-" * 50)

    for key in ["regresion_logistica", "random_forest", "xgboost"]:
        if key in eval_res:
            res = eval_res[key]
            m = res["metrics"]
            print(f"\n{res['display_name']}")
            print(f"Accuracy: {m['accuracy']*100:.2f} %")
            print(f"Precision: {m['precision']*100:.2f} %")
            print(f"Recall: {m['recall']*100:.2f} %")
            print(f"F1-Score: {m['f1_score']*100:.2f} %")
            print(f"ROC-AUC: {m['roc_auc']*100:.2f} %")

    print("\n" + "-" * 50)
    print("COMPARACIÓN")
    print("-" * 50)

    print("\nTabla comparativa (ordenada por mejor desempeño en detección y balance):")
    print(
        comp_df[
            [
                "Modelo",
                "Accuracy_Pct",
                "Precision_Pct",
                "Recall_Pct",
                "F1-Score_Pct",
                "ROC-AUC_Pct",
                "Falsos_Negativos (FN)",
            ]
        ]
        .rename(
            columns={
                "Accuracy_Pct": "Accuracy (%)",
                "Precision_Pct": "Precision (%)",
                "Recall_Pct": "Recall (%)",
                "F1-Score_Pct": "F1-Score (%)",
                "ROC-AUC_Pct": "ROC-AUC (%)",
                "Falsos_Negativos (FN)": "FN (Fallas Omitidas)",
            }
        )
        .to_string(index=False)
    )

    print(f"\nModelo seleccionado: {selected_info['nombre_modelo']}")
    print(f"Motivo: {justification}")

    print("\n" + "-" * 50)
    print("PREDICCIÓN DE FALLAS")
    print("-" * 50)

    print("\nEjemplos reales de predicciones en el conjunto de prueba (Modelo Final):")

    # Muestra balanceada de ejemplos reales con falla y sin falla
    sample_falla = df_preds[df_preds["Falla_Real"] == 1].head(3)
    sample_nofalla = df_preds[df_preds["Falla_Real"] == 0].head(2)
    sample_combined = pd.concat([sample_falla, sample_nofalla])

    for _, row in sample_combined.iterrows():
        eq_name = (
            row["Nombre_Equipo"]
            if "Nombre_Equipo" in row and pd.notna(row["Nombre_Equipo"])
            else f"Equipo ID {row['Identificador_Equipo']}"
        )
        date_val = row.get("Fecha_Observacion", "N/A")
        prob_val = row["Probabilidad_Falla_7_Dias_Pct"]
        pred_label = (
            "Falla probable en 7 días (Alerta)"
            if row["Prediccion"] == 1
            else "Operación normal (No falla)"
        )
        real_status = "Falla Real" if row["Falla_Real"] == 1 else "Sin Falla"

        print(f"\nEquipo: {eq_name} (ID: {row['Identificador_Equipo']})")
        print(f"Fecha: {date_val}")
        print(f"Estado real: {real_status}")
        print(f"Probabilidad de falla en 7 días: {prob_val:.2f} %")
        print(f"Predicción: {pred_label}")

    print("\n" + "-" * 50)
    print("ARCHIVOS GENERADOS")
    print("-" * 50)

    print(f"✓ tabla_comparacion_modelos.csv  -> {results['table_paths']['tabla_comparacion']}")
    print(f"✓ metricas_modelos.csv           -> {results['table_paths']['metricas_modelos']}")
    print(f"✓ predicciones_modelo_final.csv  -> {results['final_preds_path']}")
    print(f"✓ matrices_confusion/            -> {config.confusion_matrices_dir}")
    print(f"✓ curvas_roc/                    -> {config.roc_curves_dir}")
    print(f"✓ resultados_evaluacion/         -> {config.results_dir}")

    print("\n" + "=" * 50)
    print("FASE 7.5 - EVALUACIÓN FINALIZADA")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
