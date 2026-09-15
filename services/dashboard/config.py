"""Configuración central para la aplicación de monitoreo y predicción industrial."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class DashboardConfig:
    """Configuración de rutas, endpoints de FastAPI y umbrales de riesgo."""

    # Rutas base
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    processed_data_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    evaluation_results_dir: Path = field(default=None)
    modeling_results_dir: Path = field(default=None)
    predictions_dir: Path = field(default=None)

    # Configuración de comunicación con backend FastAPI
    api_base_url: str = field(
        default_factory=lambda: os.getenv("API_URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8502"))
    )

    # Archivos de persistencia del Dashboard
    history_file: str = "historial_predicciones.csv"
    experiments_file: str = "experimentos_streamlit.csv"

    # Títulos y metadatos
    app_title: str = "SISTEMA PREDICTIVO DE FALLAS"
    app_subtitle: str = "Área de Helados - Planta de Producción"
    app_description: str = (
        "Estima la probabilidad de falla en equipos industriales durante los "
        "próximos 7 días utilizando técnicas de Machine Learning."
    )

    # Umbrales centralizados de nivel de riesgo operativo
    risk_thresholds: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "Bajo": {
                "min": 0.00,
                "max": 0.29,
                "label": "Bajo",
                "color": "#28a745",
                "bg_color": "#d4edda",
                "icon": "🟢",
                "action": "Operación normal. Mantener rutina de lubricación e inspección programada.",
            },
            "Medio": {
                "min": 0.30,
                "max": 0.59,
                "label": "Medio",
                "color": "#ffc107",
                "bg_color": "#fff3cd",
                "icon": "🟡",
                "action": "Observación preventiva. Revisar tendencias de temperatura y vibración en próximos turnos.",
            },
            "Alto": {
                "min": 0.60,
                "max": 0.79,
                "label": "Alto",
                "color": "#fd7e14",
                "bg_color": "#ffe8d6",
                "icon": "🟠",
                "action": "Alerta de mantenimiento. Programar intervención técnica en ventana de cambio de turno.",
            },
            "Crítico": {
                "min": 0.80,
                "max": 1.00,
                "label": "Crítico",
                "color": "#dc3545",
                "bg_color": "#f8d7da",
                "icon": "🔴",
                "action": "Acción inmediata. Alta probabilidad de falla en 7 días; inspeccionar rodamientos y circuitos de frío.",
            },
        }
    )

    def __post_init__(self):
        if self.processed_data_dir is None:
            self.processed_data_dir = self.base_dir / "data" / "processed"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results"
        if self.evaluation_results_dir is None:
            self.evaluation_results_dir = self.results_dir / "evaluation"
        if self.modeling_results_dir is None:
            self.modeling_results_dir = self.results_dir / "modeling"
        if self.predictions_dir is None:
            self.predictions_dir = self.results_dir / "predictions"

        # Garantizar directorios
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.modeling_results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_results_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def get_risk_level(self, probability: float) -> Dict[str, Any]:
        """Calcula el nivel de riesgo correspondiente a una probabilidad continua [0.0 - 1.0]."""
        p = max(0.0, min(1.0, float(probability)))
        for key, conf in self.risk_thresholds.items():
            if conf["min"] <= p <= conf["max"]:
                return conf
        return self.risk_thresholds["Crítico"]
