"""Configuración para el servicio de limpieza y validación de datos."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class CleaningConfig:
    """Parámetros y rutas para el pipeline de limpieza."""

    # Rutas principales
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
        )
    )
    raw_data_dir: Path = field(default=None)
    processed_data_dir: Path = field(default=None)
    tables_dir: Path = field(default=None)
    figures_dir: Path = field(default=None)

    # Nombres de archivos
    main_dataset_name: str = "observaciones_diarias_equipo.csv"
    failures_dataset_name: str = "hechos_fallas.csv"

    # Columnas clave
    target_column: str = "Falla_En_Los_Siguientes_Siete_Dias"
    horizon_column: str = "Tiene_Horizonte_Completo_Siete_Dias"
    split_column: str = "Conjunto_Temporal"
    date_column: str = "Fecha_Observacion"
    unit_id_columns: List[str] = field(
        default_factory=lambda: ["Identificador_Equipo", "Fecha_Observacion"]
    )

    # Variables operativas
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

    # Variables categóricas
    categorical_variables: List[str] = field(
        default_factory=lambda: [
            "Tipo_Equipo",
            "Proceso",
            "Línea",
            "Criticidad",
        ]
    )

    # Identificadores a excluir como predictores
    identifier_columns: List[str] = field(
        default_factory=lambda: [
            "Identificador_Equipo",
            "Codigo_Equipo_Origen",
            "Nombre_Equipo",
        ]
    )

    # Multiplicador IQR para auditoría de atípicos
    iqr_multiplier: float = 1.5

    def __post_init__(self):
        if self.raw_data_dir is None:
            self.raw_data_dir = self.base_dir / "data" / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.base_dir / "data" / "processed"
        if self.tables_dir is None:
            self.tables_dir = self.base_dir / "tablas_preparacion"
        if self.figures_dir is None:
            self.figures_dir = self.base_dir / "figuras_preparacion"

        # Crear carpetas de salida
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
