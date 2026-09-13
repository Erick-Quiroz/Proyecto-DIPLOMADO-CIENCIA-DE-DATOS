"""Generación de características operativas y temporales sin fuga de información."""

import pandas as pd
import numpy as np
from .config import FeatureConfig


class FeatureEngineer:
    """Calcula variables operativas, ventanas móviles y deltas estrictamente históricos."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trabajar sobre una copia ordenada cronológicamente por equipo
        df_feat = df.copy()
        df_feat = df_feat.sort_values(
            by=[self.config.equipment_id_column, self.config.date_column]
        ).reset_index(drop=True)

        # Variables de ventanas móviles a 7 días y deltas a 1 día
        grouped = df_feat.groupby(self.config.equipment_id_column)
        
        for col in self.config.rolling_variables:
            # Media móvil histórica a 7 días
            df_feat[f"{col}_Media_7D"] = (
                grouped[col]
                .rolling(window=self.config.rolling_window_days, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Desviación estándar móvil a 7 días
            df_feat[f"{col}_Std_7D"] = (
                grouped[col]
                .rolling(window=self.config.rolling_window_days, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

            # Delta respecto al día anterior
            df_feat[f"{col}_Delta_1D"] = (
                grouped[col]
                .diff(1)
                .fillna(0.0)
            )

        # Variables de esfuerzo electromecánico e hidráulico
        df_feat["Carga_Electromecanica"] = df_feat["Corriente_Motor"] * df_feat["Vibracion_Equipo"]
        df_feat["Ratio_Presion_Caudal"] = df_feat["Presion_Sistema"] / (df_feat["Caudal_Proceso"] + 1e-5)

        return df_feat
