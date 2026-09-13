"""Configuración para el servicio de ingeniería de características."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


@dataclass
class FeatureConfig:
    """Parámetros de transformación, codificación y división temporal."""

    # Rutas
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    processed_data_dir: Path = field(default=None)
    tables_dir: Path = field(default=None)
    figures_dir: Path = field(default=None)

    # Columnas principales
    target_column: str = "Falla_En_Los_Siguientes_Siete_Dias"
    date_column: str = "Fecha_Observacion"
    equipment_id_column: str = "Identificador_Equipo"
    split_column: str = "Conjunto_Temporal"

    # Variables operativas base
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

    # Variables categóricas a codificar
    nominal_categoricals: List[str] = field(
        default_factory=lambda: [
            "Tipo_Equipo",
            "Proceso",
            "Línea",
        ]
    )
    ordinal_column: str = "Criticidad"
    ordinal_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "Baja": 0,
            "Media": 1,
            "Alta": 2,
        }
    )

    # Variables para cálculo de ventanas temporales e incrementales
    rolling_variables: List[str] = field(
        default_factory=lambda: [
            "Temperatura_Proceso",
            "Vibracion_Equipo",
            "Presion_Sistema",
            "Corriente_Motor",
        ]
    )
    rolling_window_days: int = 7

    # Columnas a excluir de la matriz de predictores X
    columns_to_exclude_from_x: List[str] = field(
        default_factory=lambda: [
            "Identificador_Equipo",
            "Codigo_Equipo_Origen",
            "Nombre_Equipo",
            "Fecha_Observacion",
            "Tiene_Horizonte_Completo_Siete_Dias",
            "Conjunto_Temporal",
            "Falla_En_Los_Siguientes_Siete_Dias",
        ]
    )

    def __post_init__(self):
        if self.processed_data_dir is None:
            self.processed_data_dir = self.base_dir / "data" / "processed"
        if self.tables_dir is None:
            self.tables_dir = self.base_dir / "tablas_preparacion"
        if self.figures_dir is None:
            self.figures_dir = self.base_dir / "figuras_preparacion"

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
