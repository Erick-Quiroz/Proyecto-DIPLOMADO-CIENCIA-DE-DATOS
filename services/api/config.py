"""Configuración central para el servicio de Serving con FastAPI."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class APIConfig:
    """Configuración de rutas, umbrales y persistencia de modelo activo para la API."""

    # Rutas base
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    models_dir: Path = field(default=None)
    lab_models_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    predictions_dir: Path = field(default=None)
    active_config_path: Path = field(default=None)

    # Metadatos del servicio
    service_name: str = "predictive-maintenance-api"
    service_title: str = "API de Mantenimiento Predictivo - Área de Helados"
    service_version: str = "1.0.0"
    service_description: str = (
        "API REST para la estimación de la probabilidad de falla en equipos "
        "industriales del área de helados en una ventana de 7 días (CRISP-DM Fase 7.6)."
    )

    # Modelo activo por defecto (Seleccionado en Fase 7.5 por Recall = 99.68%)
    default_active_model: str = "modelo_random_forest.joblib"

    # Umbrales centralizados de nivel de riesgo operativo
    risk_thresholds: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "Bajo": {
                "min": 0.00,
                "max": 0.29,
                "label": "Bajo",
                "action": "Operación normal. Mantener rutina de lubricación e inspección programada.",
            },
            "Medio": {
                "min": 0.30,
                "max": 0.59,
                "label": "Medio",
                "action": "Observación preventiva. Monitorear tendencias de temperatura y vibración en próximos turnos.",
            },
            "Alto": {
                "min": 0.60,
                "max": 0.79,
                "label": "Alto",
                "action": "Alerta de mantenimiento. Programar intervención técnica preventiva en cambio de turno.",
            },
            "Crítico": {
                "min": 0.80,
                "max": 1.00,
                "label": "Crítico",
                "action": "Acción inmediata. Alta probabilidad de falla en 7 días; inspeccionar rodamientos y circuito de frío.",
            },
        }
    )

    def __post_init__(self):
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.lab_models_dir is None:
            self.lab_models_dir = self.models_dir / "laboratorio"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results"
        if self.predictions_dir is None:
            self.predictions_dir = self.results_dir / "predictions"
        if self.active_config_path is None:
            self.active_config_path = self.models_dir / "active_model_config.json"

        # Garantizar directorios
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.lab_models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def get_active_model_filename(self) -> str:
        """Obtiene el nombre del archivo del modelo activo desde el archivo de configuración."""
        if self.active_config_path.exists():
            try:
                with open(self.active_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("active_model", self.default_active_model)
            except Exception:
                pass
        return self.default_active_model

    def set_active_model_filename(self, filename: str) -> None:
        """Persiste la selección del nuevo modelo activo."""
        with open(self.active_config_path, "w", encoding="utf-8") as f:
            json.dump({"active_model": filename}, f, indent=2)

    def get_risk_level(self, probability: float) -> Dict[str, Any]:
        """Calcula el nivel de riesgo y recomendación para una probabilidad continua [0.0 - 1.0]."""
        p = max(0.0, min(1.0, float(probability)))
        for key, conf in self.risk_thresholds.items():
            if conf["min"] <= p <= conf["max"]:
                return conf
        return self.risk_thresholds["Crítico"]
