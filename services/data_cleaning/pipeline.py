"""Pipeline orquestador para el servicio de limpieza y validación de datos."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .config import CleaningConfig
from .data_loader import DataLoader
from .cleaner import DataCleaner
from .leakage_guard import LeakageGuard


class CleaningPipeline:
    """Orquesta las etapas de carga, auditoría y limpieza."""

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.loader = DataLoader(config.raw_data_dir)
        self.cleaner = DataCleaner(config)
        self.guard = LeakageGuard()

    def run(self) -> Dict[str, Any]:
        # Cargar datos
        df_raw = self.loader.load_csv(self.config.main_dataset_name)

        # Revisar dimensiones y tipos
        df_audit_types = self.cleaner.audit_dimensions_and_types(df_raw)
        df_audit_types.to_csv(
            self.config.tables_dir / "auditoria_dimensiones_tipos.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Revisar duplicados
        audit_dups = self.cleaner.audit_duplicates(df_raw)
        df_audit_dups = pd.DataFrame([audit_dups])
        df_audit_dups.to_csv(
            self.config.tables_dir / "auditoria_faltantes_duplicados.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Analizar valores atípicos
        df_audit_outliers = self.cleaner.audit_outliers(df_raw)
        df_audit_outliers.to_csv(
            self.config.tables_dir / "auditoria_outliers.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Generar tabla de data leakage
        df_leakage = self.guard.generate_leakage_audit_table()
        df_leakage.to_csv(
            self.config.tables_dir / "matriz_data_leakage.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Filtrar registros sin horizonte
        df_validas, df_sin_horizonte, stats = self.cleaner.clean_and_filter(df_raw)

        # Guardar dataset limpio intermedio
        df_validas.to_csv(
            self.config.processed_data_dir / "observaciones_limpias.csv",
            index=False,
            encoding="utf-8-sig",
        )

        return {
            "status": "success",
            "df_validas": df_validas,
            "df_sin_horizonte": df_sin_horizonte,
            "stats": stats,
            "audit_types": df_audit_types,
            "audit_outliers": df_audit_outliers,
            "leakage_table": df_leakage,
        }
