"""Módulo para la carga y validación de datasets del servicio de EDA."""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger("eda.data_loader")


class DataLoader:
    """Encargado de la lectura y preparación inicial de datasets fuente en modo solo-lectura."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)

    def load_all_csvs(self) -> Dict[str, pd.DataFrame]:
        """Escanea y carga todos los archivos CSV disponibles en la carpeta de datos crudos."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"El directorio de datos crudos no existe: {self.raw_data_dir}"
            )

        csv_paths = sorted(self.raw_data_dir.glob("*.csv"))
        if not csv_paths:
            logger.warning(
                f"No se encontraron archivos CSV en: {self.raw_data_dir}"
            )
            return {}

        datos: Dict[str, pd.DataFrame] = {}
        for path in csv_paths:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
                datos[path.name] = df
                logger.info(
                    f"Cargado {path.name}: {len(df):,} filas x {df.shape[1]:,} columnas"
                )
            except Exception as e:
                logger.error(f"Error al leer {path.name}: {e}")
                raise

        return datos

    def load_specific_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Carga un archivo CSV específico por nombre."""
        path = self.raw_data_dir / filename
        if not path.exists():
            logger.warning(f"Archivo no encontrado: {path}")
            return None
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
