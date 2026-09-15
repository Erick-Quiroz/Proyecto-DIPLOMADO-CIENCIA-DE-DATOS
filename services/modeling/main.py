"""Punto de entrada CLI para el servicio de entrenamiento de modelos (CRISP-DM Fase 7.4)."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Permitir ejecución directa o como módulo
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.modeling.config import ModelingConfig
from services.modeling.pipeline import ModelingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Servicio de Modelado y Entrenamiento de Algoritmos Candidatos (CRISP-DM 7.4)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directorio con datos procesados (por defecto data/processed/)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directorio de salida para modelos (por defecto models/)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directorio de salida para tablas de resultados (por defecto results/modeling/)",
    )
    args = parser.parse_args()

    config = ModelingConfig()
    if args.data_dir:
        config.processed_data_dir = Path(args.data_dir)
    if args.models_dir:
        config.models_dir = Path(args.models_dir)
    if args.results_dir:
        config.results_dir = Path(args.results_dir)

    pipeline = ModelingPipeline(config)
    result = pipeline.run()

    print("\n" + "=" * 80)
    print("FASE 7.4: MODELADO - ENTRENAMIENTO DE MODELOS CANDIDATOS COMPLETADO")
    print("=" * 80)

    print("\n1. PARTICIONES DE DATOS UTILIZADAS:")
    shapes = result["shapes"]
    print(f" - Entrenamiento (Train) : {shapes['X_train'][0]:,d} registros, {shapes['X_train'][1]} variables | Fallas: {shapes['y_train_fallas']}")
    print(f" - Validación (Valid)    : {shapes['X_valid'][0]:,d} registros, {shapes['X_valid'][1]} variables | Fallas: {shapes['y_valid_fallas']}")
    print(f" - Prueba (Test)         : {shapes['X_test'][0]:,d} registros, {shapes['X_test'][1]} variables | Fallas: {shapes['y_test_fallas']} (Preservado intacto)")

    print("\n2. MODELOS CANDIDATOS ENTRENADOS:")
    for row in result["summary_rows"]:
        print(f"\n   * {row['nombre_modelo']} ({row['familia_algoritmo']}):")
        print(f"     - Archivo serializado: {row['archivo_modelo']}")
        print(f"     - Escalamiento interno: {'Sí (StandardScaler en Pipeline)' if row['escalamiento_interno_pipeline'] else 'No (Árboles sin escalamiento)'}")
        print(f"     - Tiempo entrenamiento: {row['tiempo_entrenamiento_seg']} segundos")
        print(f"     - Hiperparámetros base: {row['hiperparametros']}")

    print("\n3. ARTEFACTOS GENERADOS EN 'models/':")
    for key, path in result["saved_model_paths"].items():
        print(f"   - [{key}] -> {path}")

    print("\n4. RESULTADOS DE VALIDACIÓN GENERADOS EN 'results/modeling/':")
    print(f"   - Resumen de modelos      : {result['summary_path']}")
    for key, path in result["valid_paths"].items():
        print(f"   - Predicciones ({key:<12}): {path}")

    print("\n5. PRÓXIMO PASO (Fase 7.5):")
    print("   Los modelos y predicciones de validación están listos para la fase de Evaluación,")
    print("   Comparación de Métricas (ROC-AUC, PR-AUC, F1, Recall), Análisis de Costos y Selección Final.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
