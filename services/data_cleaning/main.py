"""CLI de ejecución para el servicio de limpieza y validación."""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.data_cleaning.config import CleaningConfig
from services.data_cleaning.pipeline import CleaningPipeline


def main():
    parser = argparse.ArgumentParser(description="Servicio de Limpieza y Validación de Datos")
    parser.add_argument("--raw-dir", type=str, default=None, help="Ruta a datos crudos")
    parser.add_argument("--tables-dir", type=str, default=None, help="Ruta para tablas de salida")
    args = parser.parse_args()

    config = CleaningConfig(
        raw_data_dir=Path(args.raw_dir) if args.raw_dir else None,
        tables_dir=Path(args.tables_dir) if args.tables_dir else None,
    )

    pipeline = CleaningPipeline(config)
    result = pipeline.run()

    print("[Data Cleaning] Proceso completado exitosamente.")
    print(f" - Filas iniciales: {result['stats']['filas_iniciales']}")
    print(f" - Filas sin horizonte eliminadas: {result['stats']['filas_sin_horizonte_eliminadas']}")
    print(f" - Filas válidas resultantes: {result['stats']['filas_validas_finales']}")


if __name__ == "__main__":
    main()
