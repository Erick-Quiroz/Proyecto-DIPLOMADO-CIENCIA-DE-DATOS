"""Codificación de variables categóricas garantizando cero fuga de información."""

from typing import List, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .config import FeatureConfig


class CategoricalEncoder:
    """Ajusta y transforma variables categóricas (OneHot y Ordinal) ajustadas sólo en Train."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names_ohe: List[str] = []
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame):
        # Ajustar codificador ordinal y OneHot solo en entrenamiento
        df_nominal = df_train[self.config.nominal_categoricals]
        self.ohe.fit(df_nominal)
        self.feature_names_ohe = list(self.ohe.get_feature_names_out(self.config.nominal_categoricals))
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("El codificador debe ajustarse primero con el conjunto de entrenamiento.")

        df_out = df.copy()

        # Codificar criticidad de forma ordinal
        df_out["Criticidad_Ordinal"] = df_out[self.config.ordinal_column].map(
            self.config.ordinal_mapping
        ).fillna(0).astype(int)

        # Codificar variables nominales con One-Hot
        df_nominal = df_out[self.config.nominal_categoricals]
        ohe_matrix = self.ohe.transform(df_nominal)
        df_ohe = pd.DataFrame(
            ohe_matrix,
            columns=self.feature_names_ohe,
            index=df_out.index,
        )

        # Concatenar columnas codificadas
        df_out = pd.concat([df_out, df_ohe], axis=1)

        return df_out
