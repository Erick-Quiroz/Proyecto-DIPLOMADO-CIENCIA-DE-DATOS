"""Módulo para la exportación y reporte estructurado de tablas de resultados."""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd

logger = logging.getLogger("eda.report_generator")


class ReportGenerator:
    """Encargado de persistir las tablas analíticas y el reporte de hallazgos."""

    def __init__(self, tables_dir: Path):
        self.tables_dir = Path(tables_dir)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def save_table(
        self,
        df: pd.DataFrame,
        filename: str,
        index: bool = False,
    ) -> Path:
        """Guarda un DataFrame como archivo CSV codificado en UTF-8 con BOM."""
        output_path = self.tables_dir / filename
        df.to_csv(output_path, index=index, encoding="utf-8-sig")
        logger.info(f"Tabla guardada: {output_path.name}")
        return output_path

    def save_findings_summary(
        self,
        findings: List[str],
        filename: str = "17_principales_hallazgos.csv",
    ) -> Path:
        """Exporta la lista de hallazgos sintetizados en un archivo CSV estructurado."""
        df_findings = pd.DataFrame({"Hallazgo": findings})
        return self.save_table(df_findings, filename, index=False)

    def save_all_tables(self, tables_dict: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """Guarda un diccionario completo de tablas."""
        saved_paths: Dict[str, Path] = {}
        for name, df in tables_dict.items():
            if df is not None and not df.empty:
                fname = name if name.endswith(".csv") else f"{name}.csv"
                saved_paths[name] = self.save_table(df, fname)
        return saved_paths
