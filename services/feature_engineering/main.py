"""CLI de ejecución para el servicio de ingeniería de características."""

import argparse
from pathlib import Path
import sys
import pandas as pd

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.feature_engineering.config import FeatureConfig
from services.feature_engineering.pipeline import FeaturePipeline


def main():
    parser = argparse.ArgumentParser(description="Servicio de Ingeniería de Características")
    parser.add_argument("--input-file", type=str, default=None, help="Ruta al archivo limpio de entrada")
    args = parser.parse_args()

    config = FeatureConfig()
    input_path = (
        Path(args.input_file)
        if args.input_file
        else config.processed_data_dir / "observaciones_limpias.csv"
    )

    if not input_path.exists():
        print(f"Error: Archivo de entrada no encontrado en {input_path}")
        sys.exit(1)

    df_clean = pd.read_csv(input_path, encoding="utf-8-sig")
    pipeline = FeaturePipeline(config)
    result = pipeline.run(df_clean)

    print("[Feature Engineering] Proceso completado exitosamente.")
    print(f" - Variables predictoras finales: {len(result['feature_columns'])}")
    print(f" - X_train shape: {result['X_train'].shape}")
    print(f" - X_valid shape: {result['X_valid'].shape}")
    print(f" - X_test shape: {result['X_test'].shape}")


if __name__ == "__main__":
    main()
