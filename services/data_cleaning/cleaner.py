"""Módulo de validación, auditoría y limpieza de datos."""

from typing import Dict, Tuple
import pandas as pd
import numpy as np

from .config import CleaningConfig


class DataCleaner:
    """Ejecuta la auditoría rigurosa y limpieza sin fuga de información."""

    def __init__(self, config: CleaningConfig):
        self.config = config

    def audit_dimensions_and_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # Revisar dimensiones y tipos
        audit_rows = []
        for col in df.columns:
            audit_rows.append({
                "Variable": col,
                "Tipo_Dato": str(df[col].dtype),
                "Valores_No_Nulos": int(df[col].notnull().sum()),
                "Valores_Nulos": int(df[col].isnull().sum()),
                "Pct_Nulos": round((df[col].isnull().sum() / len(df)) * 100, 2),
                "Valores_Unicos": int(df[col].nunique()),
            })
        
        audit_df = pd.DataFrame(audit_rows)
        return audit_df

    def audit_duplicates(self, df: pd.DataFrame) -> Dict[str, int]:
        # Revisar duplicados en la unidad Equipo-Dia
        total_rows = len(df)
        unit_duplicates = int(df.duplicated(subset=self.config.unit_id_columns).sum())
        exact_duplicates = int(df.duplicated().sum())

        return {
            "Total_Filas": total_rows,
            "Duplicados_Unidad_Equipo_Dia": unit_duplicates,
            "Duplicados_Filas_Exactas": exact_duplicates,
        }

    def audit_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # Analizar valores atípicos mediante IQR
        outlier_rows = []
        for col in self.config.operational_variables:
            series = df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - self.config.iqr_multiplier * iqr
            upper_bound = q3 + self.config.iqr_multiplier * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_pct = round((outlier_count / len(series)) * 100, 2)

            outlier_rows.append({
                "Variable": col,
                "Minimo": round(float(series.min()), 3),
                "Q1": round(q1, 3),
                "Mediana": round(float(series.median()), 3),
                "Q3": round(q3, 3),
                "Maximo": round(float(series.max()), 3),
                "Limite_Inferior_IQR": round(lower_bound, 3),
                "Limite_Superior_IQR": round(upper_bound, 3),
                "Cantidad_Atipicos": outlier_count,
                "Pct_Atipicos": outlier_pct,
                "Decision_Tecnica": "Conservar (Condicion Operativa Real)",
            })

        return pd.DataFrame(outlier_rows)

    def clean_and_filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        # Trabajar sobre una copia
        df_clean = df.copy()

        # Convertir fecha
        df_clean[self.config.date_column] = pd.to_datetime(df_clean[self.config.date_column])

        # Convertir variables operativas a float
        for col in self.config.operational_variables:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Separar registros sin horizonte de 7 dias
        sin_horizonte_mask = df_clean[self.config.target_column].isnull()
        df_sin_horizonte = df_clean[sin_horizonte_mask].copy()
        df_validas = df_clean[~sin_horizonte_mask].copy()

        # Asegurar tipo entero para el target valido
        df_validas[self.config.target_column] = df_validas[self.config.target_column].astype(int)

        stats = {
            "filas_iniciales": len(df_clean),
            "filas_sin_horizonte_eliminadas": len(df_sin_horizonte),
            "filas_validas_finales": len(df_validas),
            "equipos_unicos": int(df_validas["Identificador_Equipo"].nunique()),
            "fecha_minima": str(df_validas[self.config.date_column].min().date()),
            "fecha_maxima": str(df_validas[self.config.date_column].max().date()),
        }

        return df_validas, df_sin_horizonte, stats
