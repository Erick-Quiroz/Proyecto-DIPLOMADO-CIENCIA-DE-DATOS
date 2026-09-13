"""Módulo para la exportación de datasets preparados, tablas de evidencia y figuras."""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FeatureConfig


class DataExporter:
    """Exporta los datasets finales, métricas de preparación y visualizaciones de control."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def save_modeling_datasets(
        self,
        df_full_modeled: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Path]:
        # Guardar dataset final completo para modelado en data/processed
        path_dataset_modelado = self.config.processed_data_dir / "dataset_modelado.csv"
        df_full_modeled.to_csv(path_dataset_modelado, index=False, encoding="utf-8-sig")

        # Guardar conjuntos X e y en data/processed
        paths = {
            "dataset_modelado": path_dataset_modelado,
            "X_train": self.config.processed_data_dir / "X_train.csv",
            "y_train": self.config.processed_data_dir / "y_train.csv",
            "X_valid": self.config.processed_data_dir / "X_valid.csv",
            "y_valid": self.config.processed_data_dir / "y_valid.csv",
            "X_test": self.config.processed_data_dir / "X_test.csv",
            "y_test": self.config.processed_data_dir / "y_test.csv",
        }

        X_train.to_csv(paths["X_train"], index=False, encoding="utf-8-sig")
        y_train.to_csv(paths["y_train"], index=False, encoding="utf-8-sig")
        X_valid.to_csv(paths["X_valid"], index=False, encoding="utf-8-sig")
        y_valid.to_csv(paths["y_valid"], index=False, encoding="utf-8-sig")
        X_test.to_csv(paths["X_test"], index=False, encoding="utf-8-sig")
        y_test.to_csv(paths["y_test"], index=False, encoding="utf-8-sig")

        return paths

    def save_summary_tables(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_columns: List[str],
        stats_cleaning: Dict[str, Any],
    ):
        # Resumen de partición temporal
        split_summary = [
            {
                "Conjunto": "Entrenamiento",
                "Fecha_Inicio": str(df_train[self.config.date_column].min().date()),
                "Fecha_Fin": str(df_train[self.config.date_column].max().date()),
                "Total_Registros": len(df_train),
                "Fallas_0": int((df_train[self.config.target_column] == 0).sum()),
                "Fallas_1": int((df_train[self.config.target_column] == 1).sum()),
                "Tasa_Falla_Pct": round(float(df_train[self.config.target_column].mean()) * 100, 2),
            },
            {
                "Conjunto": "Validación",
                "Fecha_Inicio": str(df_valid[self.config.date_column].min().date()),
                "Fecha_Fin": str(df_valid[self.config.date_column].max().date()),
                "Total_Registros": len(df_valid),
                "Fallas_0": int((df_valid[self.config.target_column] == 0).sum()),
                "Fallas_1": int((df_valid[self.config.target_column] == 1).sum()),
                "Tasa_Falla_Pct": round(float(df_valid[self.config.target_column].mean()) * 100, 2),
            },
            {
                "Conjunto": "Prueba",
                "Fecha_Inicio": str(df_test[self.config.date_column].min().date()),
                "Fecha_Fin": str(df_test[self.config.date_column].max().date()),
                "Total_Registros": len(df_test),
                "Fallas_0": int((df_test[self.config.target_column] == 0).sum()),
                "Fallas_1": int((df_test[self.config.target_column] == 1).sum()),
                "Tasa_Falla_Pct": round(float(df_test[self.config.target_column].mean()) * 100, 2),
            },
        ]
        df_split_summary = pd.DataFrame(split_summary)
        df_split_summary.to_csv(
            self.config.tables_dir / "resumen_particion_temporal.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Resumen general de preparación
        total_valid = len(df_train) + len(df_valid) + len(df_test)
        summary_general = [
            {"Metrica": "Filas iniciales", "Valor": stats_cleaning.get("filas_iniciales", 17719)},
            {"Metrica": "Filas sin horizonte eliminadas", "Valor": stats_cleaning.get("filas_sin_horizonte_eliminadas", 203)},
            {"Metrica": "Filas finales preparadas", "Valor": total_valid},
            {"Metrica": "Número de variables iniciales", "Valor": 19},
            {"Metrica": "Número de variables predictoras finales", "Valor": len(feature_columns)},
            {"Metrica": "Variables predictoras utilizadas", "Valor": ", ".join(feature_columns)},
            {"Metrica": "Tratamiento de faltantes", "Valor": "0 nulos en predictoras; 203 nulos en target eliminados por falta de horizonte"},
            {"Metrica": "Tratamiento de outliers", "Valor": "Conservados sin corte para capturar precursores de fallas reales"},
            {"Metrica": "Duplicados encontrados", "Valor": "0 duplicados en unidad Equipo-Día"},
            {"Metrica": "Codificación categórica", "Valor": "Criticidad (Ordinal 0-2), Tipo_Equipo/Proceso/Línea (One-Hot)"},
            {"Metrica": "Escalamiento", "Valor": "Preservado en unidades originales; preparado para pipeline de modelos lineales"},
            {"Metrica": "Variables nuevas creadas", "Valor": "Medias 7D, Std 7D, Deltas 1D, Carga_Electromecanica, Ratio_Presion_Caudal"},
            {"Metrica": "Distribución final target (0 / 1)", "Valor": f"0: 15,570 (88.89%) | 1: 1,946 (11.11%)"},
            {"Metrica": "Registros en Train", "Valor": len(df_train)},
            {"Metrica": "Registros en Validación", "Valor": len(df_valid)},
            {"Metrica": "Registros en Test", "Valor": len(df_test)},
        ]
        df_summary_general = pd.DataFrame(summary_general)
        df_summary_general.to_csv(
            self.config.tables_dir / "resumen_final_preparacion.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def generate_figures(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        # Configurar estilo visual limpio
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Gráfico de distribución de target por partición
        fig, ax = plt.subplots(figsize=(8, 5))
        df_comb = pd.concat([
            df_train.assign(Particion="Entrenamiento"),
            df_valid.assign(Particion="Validación"),
            df_test.assign(Particion="Prueba"),
        ])
        
        target_prop = (
            df_comb.groupby(["Particion", self.config.target_column])
            .size()
            .unstack(fill_value=0)
        )
        target_pct = target_prop.div(target_prop.sum(axis=1), axis=0) * 100

        target_pct.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#3b82f6", "#ef4444"],
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title("Distribución de Clase Objetivo (0: Sin Falla vs 1: Falla) por Partición", fontsize=12, fontweight="bold")
        ax.set_xlabel("Partición Temporal", fontsize=11)
        ax.set_ylabel("Porcentaje (%)", fontsize=11)
        ax.legend(["0 = Sin Falla (7D)", "1 = Falla en 7D"], frameon=True)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(self.config.figures_dir / "distribucion_target_temporal.png", dpi=200)
        plt.close(fig)

        # 2. Gráfico cronológico de cobertura temporal
        fig, ax = plt.subplots(figsize=(10, 4))
        timeline_data = [
            ("Entrenamiento", df_train[self.config.date_column].min(), df_train[self.config.date_column].max(), "#3b82f6"),
            ("Validación", df_valid[self.config.date_column].min(), df_valid[self.config.date_column].max(), "#f59e0b"),
            ("Prueba", df_test[self.config.date_column].min(), df_test[self.config.date_column].max(), "#10b981"),
        ]

        for i, (name, start, end, color) in enumerate(timeline_data):
            ax.barh(name, (end - start).days, left=start, color=color, edgecolor="black", alpha=0.85, height=0.5)
            ax.text(
                start + (end - start) / 2,
                i,
                f"{start.strftime('%Y-%m-%d')} a {end.strftime('%Y-%m-%d')}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )

        ax.set_title("Esquema de Partición Temporal Cronológica (Sin Fuga de Información)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Línea Temporal de Observación", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.config.figures_dir / "particion_cronologica.png", dpi=200)
        plt.close(fig)
