"""Módulo de partición temporal cronológica de datasets y separación de X e y."""

from typing import Dict, Tuple, List
import pandas as pd

from .config import FeatureConfig


class TemporalSplitter:
    """Realiza la partición temporal estricta y separación de matrices X e y."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def split_by_chronology(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Partición cronológica según Conjunto_Temporal
        df_train = df[df[self.config.split_column] == "Entrenamiento"].copy()
        df_valid = df[df[self.config.split_column] == "Validacion"].copy()
        df_test = df[df[self.config.split_column] == "Prueba"].copy()

        return df_train, df_valid, df_test

    def extract_x_y(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # Extraer variables predictoras X y variable objetivo y
        X = df[feature_columns].copy()
        y = df[self.config.target_column].astype(int).copy()
        return X, y

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        # Determinar columnas predictoras finales excluyendo identificadores, fechas y metadatos
        exclude = set(self.config.columns_to_exclude_from_x).union(
            set(self.config.nominal_categoricals)
        ).union({self.config.ordinal_column})

        feature_cols = [col for col in df.columns if col not in exclude]
        return feature_cols
