"""Módulo de partición temporal cronológica de datasets y separación de X e y con purga de leakage."""

from typing import Dict, Tuple, List, Any
import pandas as pd

from .config import FeatureConfig


class TemporalSplitter:
    """Realiza la partición temporal estricta con ventana de purga y separación de matrices X e y."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def split_by_chronology(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide el dataset en Train, Valid y Test aplicando ventanas de purga de 7 días."""
        df_work = df.copy()
        df_work[self.config.date_column] = pd.to_datetime(df_work[self.config.date_column])

        val_start = pd.to_datetime(self.config.val_start_date)
        test_start = pd.to_datetime(self.config.test_start_date)
        horizon = pd.Timedelta(days=self.config.target_horizon_days)

        # 1. Regla Train: último target de Train < inicio de Validation
        train_mask = (df_work[self.config.date_column] + horizon) < val_start
        df_train = df_work[train_mask].copy()

        # 2. Regla Validation: inicio en val_start y último target de Validation < inicio de Test
        valid_mask = (df_work[self.config.date_column] >= val_start) & (
            (df_work[self.config.date_column] + horizon) < test_start
        )
        df_valid = df_work[valid_mask].copy()

        # 3. Regla Test: inicio en test_start
        test_mask = df_work[self.config.date_column] >= test_start
        df_test = df_work[test_mask].copy()

        # 4. Verificación automática de temporal leakage
        self.validate_temporal_leakage(df_train, df_valid, df_test)

        return df_train, df_valid, df_test

    def assign_split_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asigna etiquetas de partición y purga sobre el dataset consolidado sin eliminar filas."""
        df_out = df.copy()
        df_out[self.config.date_column] = pd.to_datetime(df_out[self.config.date_column])

        val_start = pd.to_datetime(self.config.val_start_date)
        test_start = pd.to_datetime(self.config.test_start_date)
        horizon = pd.Timedelta(days=self.config.target_horizon_days)

        labels = pd.Series("Purga", index=df_out.index)

        train_mask = (df_out[self.config.date_column] + horizon) < val_start
        valid_mask = (df_out[self.config.date_column] >= val_start) & (
            (df_out[self.config.date_column] + horizon) < test_start
        )
        test_mask = df_out[self.config.date_column] >= test_start

        labels[train_mask] = "Entrenamiento"
        labels[valid_mask] = "Validacion"
        labels[test_mask] = "Prueba"

        df_out[self.config.split_column] = labels
        return df_out

    def validate_temporal_leakage(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Ejecuta la verificación programática del horizonte de predicción y fronteras temporales."""
        horizon = pd.Timedelta(days=self.config.target_horizon_days)
        date_col = self.config.date_column

        train_obs_max = df_train[date_col].max()
        train_target_max = train_obs_max + horizon

        val_obs_min = df_valid[date_col].min()
        val_obs_max = df_valid[date_col].max()
        val_target_max = val_obs_max + horizon

        test_obs_min = df_test[date_col].min()

        cond_train_valid = train_target_max < val_obs_min
        cond_valid_test = val_target_max < test_obs_min

        status_train_val = "PASS" if cond_train_valid else "FAIL"
        status_val_test = "PASS" if cond_valid_test else "FAIL"

        print("\n" + "=" * 50)
        print("VALIDACIÓN DE LEAKAGE TEMPORAL")
        print("=" * 50)
        print("\nTrain:")
        print(f"Observación máxima: {train_obs_max.strftime('%Y-%m-%d')}")
        print(f"Target máximo: {train_target_max.strftime('%Y-%m-%d')}")
        print("\nValidation:")
        print(f"Observación mínima: {val_obs_min.strftime('%Y-%m-%d')}")
        print(f"Observación máxima: {val_obs_max.strftime('%Y-%m-%d')}")
        print(f"Target máximo: {val_target_max.strftime('%Y-%m-%d')}")
        print("\nTest:")
        print(f"Observación mínima: {test_obs_min.strftime('%Y-%m-%d')}")
        print(f"\nCondición Train → Validation:")
        print(f"{status_train_val}")
        print(f"\nCondición Validation → Test:")
        print(f"{status_val_test}")
        print("=" * 50 + "\n")

        if not cond_train_valid or not cond_valid_test:
            raise ValueError(
                f"FALLO DE VALIDACIÓN: Temporal Leakage detectado. Train->Valid: {status_train_val}, Valid->Test: {status_val_test}"
            )

        return {
            "train_obs_max": train_obs_max.strftime("%Y-%m-%d"),
            "train_target_max": train_target_max.strftime("%Y-%m-%d"),
            "val_obs_min": val_obs_min.strftime("%Y-%m-%d"),
            "val_obs_max": val_obs_max.strftime("%Y-%m-%d"),
            "val_target_max": val_target_max.strftime("%Y-%m-%d"),
            "test_obs_min": test_obs_min.strftime("%Y-%m-%d"),
            "status_train_valid": status_train_val,
            "status_valid_test": status_val_test,
        }

    def extract_x_y(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extrae variables predictoras X y variable objetivo y."""
        X = df[feature_columns].copy()
        y = df[self.config.target_column].astype(int).copy()
        return X, y

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Determina columnas predictoras finales excluyendo identificadores, fechas y metadatos."""
        exclude = set(self.config.columns_to_exclude_from_x).union(
            set(self.config.nominal_categoricals)
        ).union({self.config.ordinal_column})

        feature_cols = [col for col in df.columns if col not in exclude]
        return feature_cols

