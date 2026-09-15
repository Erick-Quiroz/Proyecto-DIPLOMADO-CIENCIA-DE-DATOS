"""Definición y factoría de modelos candidatos para clasificación binaria."""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from services.modeling.config import ModelingConfig


class CandidateModelsFactory:
    """Factoría para la construcción y parametrización de los modelos candidatos."""

    def __init__(self, config: ModelingConfig):
        self.config = config

    def build_logistic_regression(self) -> Pipeline:
        """Construye el pipeline de Regresión Logística con escalamiento estándar encapsulado."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**self.config.logistic_regression_params)),
            ]
        )

    def build_random_forest(self) -> RandomForestClassifier:
        """Construye el estimador Random Forest sin escalamiento."""
        return RandomForestClassifier(**self.config.random_forest_params)

    def build_xgboost(self) -> XGBClassifier:
        """Construye el estimador XGBoost sin escalamiento."""
        return XGBClassifier(**self.config.xgboost_params)

    def build_all(self) -> Dict[str, Dict[str, Any]]:
        """Retorna el catálogo completo de modelos candidatos listos para entrenamiento."""
        return {
            "regresion_logistica": {
                "display_name": "Regresión Logística",
                "family": "Modelo Lineal Generalizado",
                "estimator": self.build_logistic_regression(),
                "file_name": self.config.logistic_regression_model_file,
                "valid_pred_file": self.config.lr_valid_predictions_file,
                "has_internal_scaler": True,
                "params": self.config.logistic_regression_params,
            },
            "random_forest": {
                "display_name": "Random Forest",
                "family": "Ensemble (Bagging)",
                "estimator": self.build_random_forest(),
                "file_name": self.config.random_forest_model_file,
                "valid_pred_file": self.config.rf_valid_predictions_file,
                "has_internal_scaler": False,
                "params": self.config.random_forest_params,
            },
            "xgboost": {
                "display_name": "XGBoost",
                "family": "Ensemble (Boosting)",
                "estimator": self.build_xgboost(),
                "file_name": self.config.xgboost_model_file,
                "valid_pred_file": self.config.xgb_valid_predictions_file,
                "has_internal_scaler": False,
                "params": self.config.xgboost_params,
            },
        }
