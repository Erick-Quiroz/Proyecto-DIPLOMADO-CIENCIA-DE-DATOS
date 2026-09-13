"""Script principal reproducible para la Fase 7.3: Preparación de Datos."""

import os
import sys
from pathlib import Path
import pandas as pd

# Configurar ruta base
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from services.data_cleaning.config import CleaningConfig
from services.data_cleaning.pipeline import CleaningPipeline
from services.feature_engineering.config import FeatureConfig
from services.feature_engineering.pipeline import FeaturePipeline


def run_preparation():
    # Cargar configuraciones
    cleaning_config = CleaningConfig(base_dir=BASE_DIR)
    feature_config = FeatureConfig(base_dir=BASE_DIR)

    # 1. Ejecutar servicio de limpieza y auditoría
    cleaning_pipeline = CleaningPipeline(cleaning_config)
    cleaning_result = cleaning_pipeline.run()

    df_validas = cleaning_result["df_validas"]
    stats_cleaning = cleaning_result["stats"]

    # 2. Ejecutar servicio de ingeniería de características, codificación y split
    feature_pipeline = FeaturePipeline(feature_config)
    feature_result = feature_pipeline.run(df_validas, stats_cleaning=stats_cleaning)

    df_full = feature_result["df_full"]
    feature_columns = feature_result["feature_columns"]
    X_train = feature_result["X_train"]
    y_train = feature_result["y_train"]
    X_valid = feature_result["X_valid"]
    y_valid = feature_result["y_valid"]
    X_test = feature_result["X_test"]
    y_test = feature_result["y_test"]

    # 3. Presentar resumen formal requerido
    print("\n" + "=" * 70)
    print("Dataset final preparado para el modelado")
    print("=" * 70)
    print(f"\n1. DIMENSIONES DEL DATASET:")
    print(f" - Filas totales preparadas: {len(df_full):,}")
    print(f" - Columnas totales: {df_full.shape[1]}")
    print(f" - Variables predictoras finales en X: {len(feature_columns)}")

    print(f"\n2. LISTA DE COLUMNAS PREDICTORAS FINALES ({len(feature_columns)}):")
    for i, col in enumerate(feature_columns, 1):
        print(f"   {i:2d}. {col}")

    print(f"\n3. PRIMERAS FILAS DEL DATASET (Muestra de variables predictoras):")
    sample_cols = feature_columns[:6] + [feature_config.target_column]
    print(df_full[sample_cols].head(5).to_string())

    print(f"\n4. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (Falla_En_Los_Siguientes_Siete_Dias):")
    target_counts = df_full[feature_config.target_column].value_counts()
    target_pcts = df_full[feature_config.target_column].value_counts(normalize=True) * 100
    for val in [0, 1]:
        print(f"   Clase {val} ({'Sin Falla en 7D' if val == 0 else 'Falla en 7D'}): {target_counts[val]:,d} registros ({target_pcts[val]:.2f}%)")

    print(f"\n5. PARTICIÓN CRONOLÓGICA (TRAIN / VALIDACIÓN / TEST):")
    print(f"   - Entrenamiento (Train) : {len(X_train):,d} registros ({len(X_train)/len(df_full)*100:.2f}%) | Fallas: {y_train.sum():,d} ({y_train.mean()*100:.2f}%)")
    print(f"   - Validación (Valid)    : {len(X_valid):,d} registros ({len(X_valid)/len(df_full)*100:.2f}%) | Fallas: {y_valid.sum():,d} ({y_valid.mean()*100:.2f}%)")
    print(f"   - Prueba (Test)         : {len(X_test):,d} registros ({len(X_test)/len(df_full)*100:.2f}%) | Fallas: {y_test.sum():,d} ({y_test.mean()*100:.2f}%)")

    print(f"\n6. UBICACIÓN DE LOS ARCHIVOS GENERADOS:")
    print(f"   - Dataset consolidado: {feature_config.processed_data_dir / 'dataset_modelado.csv'}")
    print(f"   - X_train            : {feature_config.processed_data_dir / 'X_train.csv'}")
    print(f"   - y_train            : {feature_config.processed_data_dir / 'y_train.csv'}")
    print(f"   - X_valid            : {feature_config.processed_data_dir / 'X_valid.csv'}")
    print(f"   - y_valid            : {feature_config.processed_data_dir / 'y_valid.csv'}")
    print(f"   - X_test             : {feature_config.processed_data_dir / 'X_test.csv'}")
    print(f"   - y_test             : {feature_config.processed_data_dir / 'y_test.csv'}")
    print(f"   - Tablas de evidencia: {feature_config.tables_dir}")
    print(f"   - Figuras de control : {feature_config.figures_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_preparation()
