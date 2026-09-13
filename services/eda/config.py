"""Configuración centralizada para el servicio de EDA."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class EDAConfig:
    """Configuración de rutas y parámetros para el pipeline de EDA."""

    # Rutas base
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    raw_data_dir: Path = field(default=None)
    figures_dir: Path = field(default=None)
    tables_dir: Path = field(default=None)

    # Parámetros analíticos
    target_column: str = "Falla_En_Los_Siguientes_Siete_Dias"
    pre_failure_window_days: int = 21
    max_categorical_cardinality: int = 20
    iqr_multiplier: float = 1.5
    figure_dpi: int = 200

    operational_variables: List[str] = field(
        default_factory=lambda: [
            "Temperatura_Proceso",
            "Vibracion_Equipo",
            "Presion_Sistema",
            "Corriente_Motor",
            "Caudal_Proceso",
            "Nivel_Sistema",
            "Velocidad_Accionamiento",
            "Horas_Operacion",
        ]
    )

    leakage_risk_keywords: List[str] = field(
        default_factory=lambda: [
            "falla_registrada",
            "tipo_falla",
            "severidad",
            "fecha_falla",
            "vida_útil",
            "vida_util",
            "probabilidad_falla",
        ]
    )

    def __post_init__(self):
        if self.raw_data_dir is None:
            raw_env = os.getenv("DATA_RAW_DIR")
            self.raw_data_dir = (
                Path(raw_env) if raw_env else self.base_dir / "data" / "raw"
            )

        if self.figures_dir is None:
            fig_env = os.getenv("EDA_FIGURES_DIR")
            self.figures_dir = (
                Path(fig_env)
                if fig_env
                else self.base_dir / "reports" / "figures" / "eda"
            )

        if self.tables_dir is None:
            tab_env = os.getenv("EDA_TABLES_DIR")
            self.tables_dir = (
                Path(tab_env)
                if tab_env
                else self.base_dir / "reports" / "tables" / "eda"
            )

        # Crear directorios de salida si no existen
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
