"""Módulo para la generación y exportación de gráficos de alta resolución."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("eda.visualizer")


class FigureExporter:
    """Generador estandarizado de visualizaciones para el análisis exploratorio."""

    def __init__(self, figures_dir: Path, dpi: int = 200):
        self.figures_dir = Path(figures_dir)
        self.dpi = dpi
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_target_distribution(
        self,
        counts: Dict[str, int],
        filename: str = "01_variable_objetivo.png",
    ) -> Path:
        """Grafica la distribución de clases de la variable objetivo."""
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(counts.keys())
        values = list(counts.values())

        ax.bar(labels, values, color=["#2b5c8f", "#d9534f", "#f0ad4e"])
        ax.set_title("Distribución de la variable objetivo", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cantidad de observaciones", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=15)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_histogram(
        self,
        series: pd.Series,
        col_name: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Grafica el histograma de una variable operativa continua."""
        fname = filename or f"02_hist_{col_name}.png"
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(series.dropna(), bins=30, color="#3498db", edgecolor="#2980b9", alpha=0.85)
        ax.set_title(f"Distribución de {col_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col_name, fontsize=10)
        ax.set_ylabel("Frecuencia", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = self.figures_dir / fname
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        filename: str = "03_matriz_correlacion.png",
    ) -> Path:
        """Grafica la matriz de correlación como mapa de calor."""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=90, ha="right")
        ax.set_yticklabels(corr_matrix.index)
        ax.set_title("Matriz de correlación de variables operativas", fontsize=12, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, label="Correlación")
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_failures_by_month(
        self,
        df_monthly: pd.DataFrame,
        filename: str = "04_fallas_por_mes.png",
    ) -> Path:
        """Grafica la serie mensual de fallas."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            df_monthly["Mes"],
            df_monthly["Cantidad_Fallas"],
            marker="o",
            color="#e74c3c",
            linewidth=2,
            markersize=5,
        )
        ax.set_title("Cantidad de fallas por mes", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mes", fontsize=10)
        ax.set_ylabel("Cantidad de fallas", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.xticks(rotation=60)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_top_equipment_failures(
        self,
        df_top: pd.DataFrame,
        filename: str = "05_fallas_por_equipo.png",
    ) -> Path:
        """Grafica las fallas acumuladas por equipo (top)."""
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            df_top["Identificador_Equipo"].astype(str),
            df_top["Cantidad_Fallas"],
            color="#2c3e50",
        )
        ax.set_title("Equipos con mayor cantidad de fallas", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cantidad de fallas", fontsize=10)
        ax.set_ylabel("Identificador del equipo", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_failures_by_process(
        self,
        df_process: pd.DataFrame,
        filename: str = "06_fallas_por_proceso.png",
    ) -> Path:
        """Grafica la distribución de fallas agrupadas por proceso."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            df_process["Proceso"].astype(str),
            df_process["Cantidad_Fallas"],
            color="#8e44ad",
        )
        ax.set_title("Distribución de fallas por proceso", fontsize=12, fontweight="bold")
        ax.set_xlabel("Proceso", fontsize=10)
        ax.set_ylabel("Cantidad de fallas", fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_boxplots_by_process(
        self,
        df_obs: pd.DataFrame,
        variable: str,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Genera diagrama de caja para una variable agrupada por proceso."""
        if variable not in df_obs.columns or "Proceso" not in df_obs.columns:
            return None

        fname = filename or f"07_boxplot_{variable}_proceso.png"
        fig, ax = plt.subplots(figsize=(11, 5))

        df_obs.boxplot(column=variable, by="Proceso", rot=35, ax=ax)
        ax.set_title(f"{variable} según proceso", fontsize=12, fontweight="bold")
        ax.set_xlabel("Proceso", fontsize=10)
        ax.set_ylabel(variable, fontsize=10)
        plt.suptitle("")
        plt.tight_layout()

        output_path = self.figures_dir / fname
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path

    def plot_pre_failure_evolution(
        self,
        df_temporal: pd.DataFrame,
        variable: str,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Grafica la trayectoria promedio de una variable en los 21 días previos a la falla."""
        if df_temporal.empty or variable not in df_temporal.columns:
            return None

        fname = filename or f"08_evolucion_previa_{variable}.png"
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            df_temporal["Dias_Antes_Falla"],
            df_temporal[variable],
            marker="o",
            color="#d35400",
            linewidth=2,
        )
        ax.invert_xaxis()
        ax.set_xlabel("Días antes de la falla", fontsize=10)
        ax.set_ylabel(variable, fontsize=10)
        ax.set_title(
            f"Evolución de {variable} durante los 21 días previos a la falla",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        output_path = self.figures_dir / fname
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figura guardada: {output_path.name}")
        return output_path
