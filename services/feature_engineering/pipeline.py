"""Pipeline orquestador para ingeniería de características, codificación y partición."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .config import FeatureConfig
from .encoders import CategoricalEncoder
from .features import FeatureEngineer
from .splitter import TemporalSplitter
from .exporter import DataExporter


class FeaturePipeline:
    """Orquesta la ingeniería de features, codificación, partición y exportación."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.encoder = CategoricalEncoder(config)
        self.engineer = FeatureEngineer(config)
        self.splitter = TemporalSplitter(config)
        self.exporter = DataExporter(config)

    def run(self, df_clean: pd.DataFrame, stats_cleaning: Dict[str, Any] = None) -> Dict[str, Any]:
        if stats_cleaning is None:
            stats_cleaning = {}

        # Generar variables históricas y temporales
        df_featured = self.engineer.create_features(df_clean)

        # Partición cronológica inicial
        df_train_raw, df_valid_raw, df_test_raw = self.splitter.split_by_chronology(df_featured)

        # Ajustar codificador categórico únicamente en Train
        self.encoder.fit(df_train_raw)

        # Transformar conjuntos de forma independiente
        df_train = self.encoder.transform(df_train_raw)
        df_valid = self.encoder.transform(df_valid_raw)
        df_test = self.encoder.transform(df_test_raw)
        
        # Asignar etiquetas actualizadas de partición y purga al dataset consolidado
        df_full = self.encoder.transform(df_featured)
        df_full = self.splitter.assign_split_labels(df_full)

        # Determinar columnas predictoras finales
        feature_columns = self.splitter.get_feature_columns(df_train)

        # Extraer matrices X e y
        X_train, y_train = self.splitter.extract_x_y(df_train, feature_columns)
        X_valid, y_valid = self.splitter.extract_x_y(df_valid, feature_columns)
        X_test, y_test = self.splitter.extract_x_y(df_test, feature_columns)

        # Guardar datasets preparados y matrices
        saved_paths = self.exporter.save_modeling_datasets(
            df_full_modeled=df_full,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )

        # Guardar tablas y figuras
        self.exporter.save_summary_tables(
            df_train=df_train,
            df_valid=df_valid,
            df_test=df_test,
            feature_columns=feature_columns,
            stats_cleaning=stats_cleaning,
        )

        self.exporter.generate_figures(
            df_train=df_train,
            df_valid=df_valid,
            df_test=df_test,
        )

        return {
            "status": "success",
            "saved_paths": saved_paths,
            "feature_columns": feature_columns,
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
            "df_full": df_full,
        }
