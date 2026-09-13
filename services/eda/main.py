"""Punto de entrada CLI para el servicio desacoplado de EDA."""

import argparse
import logging
import sys
from pathlib import Path

# Permitir ejecución tanto como script directo como módulo del paquete
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.eda.config import EDAConfig
from services.eda.pipeline import EDAPipeline


def setup_logging(verbose: bool = False):
    """Configura el formato y nivel del registro de eventos."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Servicio de Análisis Exploratorio de Datos (EDA) para MLOps"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help="Ruta al directorio de datos crudos (por defecto: data/raw)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Ruta al directorio de figuras de salida (por defecto: reports/figures/eda)",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=None,
        help="Ruta al directorio de tablas de salida (por defecto: reports/tables/eda)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolución DPI para exportación de figuras (por defecto: 200)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Activar registro detallado (DEBUG)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = EDAConfig(
        raw_data_dir=Path(args.raw_data_dir) if args.raw_data_dir else None,
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        tables_dir=Path(args.tables_dir) if args.tables_dir else None,
        figure_dpi=args.dpi,
    )

    pipeline = EDAPipeline(config=config)
    result = pipeline.run()

    if result.get("status") == "success":
        print("\n[EDA Service] Ejecución finalizada correctamente.")
        print(f" - Datasets procesados: {len(result['datasets_loaded'])}")
        print(f" - Tablas generadas: {result['tables_generated']}")
        print(f" - Figuras generadas: {result['figures_generated']}")
        print(f" - Hallazgos identificados: {len(result['findings'])}")
    else:
        print(f"\n[EDA Service] Aviso: {result.get('message')}")


if __name__ == "__main__":
    main()
