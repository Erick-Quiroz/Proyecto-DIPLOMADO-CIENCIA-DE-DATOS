"""Carga segura y lectura de datasets crudos."""

from pathlib import Path
import pandas as pd


class DataLoader:
    """Cargador de datos solo-lectura."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Carga un dataset CSV retornando una copia independiente."""
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado en {filepath}")
        
        # Cargar datos
        df = pd.read_csv(filepath, encoding="utf-8-sig", low_memory=False)
        return df.copy()
