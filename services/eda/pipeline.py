"""Orquestador principal del pipeline y servicio de EDA."""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from services.eda.config import EDAConfig
from services.eda.data_loader import DataLoader
from services.eda.visualizer import FigureExporter
from services.eda.report_generator import ReportGenerator
from services.eda.analyzers import (
    InventoryAnalyzer,
    StatisticalAnalyzer,
    TargetAnalyzer,
    OperationalAnalyzer,
    TemporalAnalyzer,
    DataLeakageAuditor,
)

logger = logging.getLogger("eda.pipeline")


class EDAPipeline:
    """Orquesta la ejecución secuencial, desacoplada y reproducible del análisis exploratorio."""

    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()
        self.data_loader = DataLoader(self.config.raw_data_dir)
        self.visualizer = FigureExporter(self.config.figures_dir, dpi=self.config.figure_dpi)
        self.reporter = ReportGenerator(self.config.tables_dir)

        # Registro de analizadores modulares
        self.analyzers = [
            InventoryAnalyzer(self.config),
            StatisticalAnalyzer(self.config),
            TargetAnalyzer(self.config),
            OperationalAnalyzer(self.config),
            TemporalAnalyzer(self.config),
            DataLeakageAuditor(self.config),
        ]

    def run(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de EDA de forma modular."""
        logger.info("=== Iniciando Pipeline de EDA ===")
        logger.info(f"Directorio fuente: {self.config.raw_data_dir}")
        logger.info(f"Directorio de figuras: {self.config.figures_dir}")
        logger.info(f"Directorio de tablas: {self.config.tables_dir}")

        # 1. Carga de datos
        datasets = self.data_loader.load_all_csvs()
        if not datasets:
            logger.warning(
                "No se encontraron datasets para analizar. "
                "Coloque los archivos CSV en 'data/raw' para ejecutar el análisis completo."
            )
            return {
                "status": "warning",
                "message": "Directorio data/raw sin archivos CSV.",
                "tables_generated": 0,
                "figures_generated": 0,
                "findings": [],
            }

        all_tables: Dict[str, pd.DataFrame] = {}
        all_findings: List[str] = []
        pipeline_metadata: Dict[str, Any] = {}

        # 2. Ejecución de analizadores
        for analyzer in self.analyzers:
            analyzer_name = analyzer.__class__.__name__
            logger.info(f"Ejecutando: {analyzer_name}...")
            res = analyzer.analyze(datasets)
            all_tables.update(res.tables)
            all_findings.extend(res.findings)
            pipeline_metadata.update(res.metadata)

        # 3. Generación y exportación de figuras
        fig_count = self._generate_visualizations(datasets, pipeline_metadata)

        # 4. Guardar tablas estructuradas y reporte de hallazgos
        saved_tables = self.reporter.save_all_tables(all_tables)
        if all_findings:
            self.reporter.save_findings_summary(all_findings)

        logger.info("=== Pipeline de EDA Finalizado Exitosamente ===")
        logger.info(f"Tablas exportadas: {len(saved_tables)}")
        logger.info(f"Figuras exportadas: {fig_count}")
        logger.info(f"Hallazgos consolidados: {len(all_findings)}")

        return {
            "status": "success",
            "datasets_loaded": list(datasets.keys()),
            "tables_generated": len(saved_tables),
            "figures_generated": fig_count,
            "findings": all_findings,
        }

    def _generate_visualizations(
        self,
        datasets: Dict[str, pd.DataFrame],
        metadata: Dict[str, Any],
    ) -> int:
        """Genera el conjunto estándar de figuras."""
        fig_count = 0
        obs = datasets.get("observaciones_diarias_equipo.csv")

        # 1. Distribución del target
        target_counts = metadata.get("target_counts")
        if target_counts:
            self.visualizer.plot_target_distribution(target_counts)
            fig_count += 1

        if obs is not None:
            op_vars = [c for c in self.config.operational_variables if c in obs.columns]

            # 2. Histogramas de variables operativas
            for col in op_vars:
                s = pd.to_numeric(obs[col], errors="coerce").dropna()
                if len(s) > 0:
                    self.visualizer.plot_histogram(s, col)
                    fig_count += 1

            # 3. Matriz de correlación
            corr_matrix = metadata.get("correlation_matrix")
            if corr_matrix is not None:
                self.visualizer.plot_correlation_matrix(corr_matrix)
                fig_count += 1

            # 4. Diagramas de caja por proceso
            if "Proceso" in obs.columns:
                for col in ["Temperatura_Proceso", "Vibracion_Equipo", "Presion_Sistema", "Corriente_Motor"]:
                    if col in obs.columns:
                        self.visualizer.plot_boxplots_by_process(obs, col)
                        fig_count += 1

        # 5. Fallas por mes
        df_mes = metadata.get("fallas_mes")
        if df_mes is not None and not df_mes.empty:
            self.visualizer.plot_failures_by_month(df_mes)
            fig_count += 1

        # 6. Fallas por equipo (Top 15)
        df_eq = metadata.get("fallas_equipo")
        if df_eq is not None and not df_eq.empty:
            top_eq = df_eq.head(15).sort_values("Cantidad_Fallas")
            self.visualizer.plot_top_equipment_failures(top_eq)
            fig_count += 1

        # 7. Fallas por proceso
        df_proc = metadata.get("fallas_proceso")
        if df_proc is not None and not df_proc.empty:
            self.visualizer.plot_failures_by_process(df_proc)
            fig_count += 1

        # 8. Evolución previa a la falla (21 días)
        df_prev = metadata.get("resumen_temporal_previo")
        if df_prev is not None and not df_prev.empty:
            op_vars_prev = [c for c in self.config.operational_variables if c in df_prev.columns]
            for col in op_vars_prev:
                self.visualizer.plot_pre_failure_evolution(df_prev, col)
                fig_count += 1

        return fig_count
