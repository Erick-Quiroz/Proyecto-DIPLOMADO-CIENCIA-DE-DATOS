"""Módulo de validación de esquemas, tipos y completitud de datasets crudos."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class DatasetSchema:
    """Definición del esquema esperado para un dataset."""
    filename: str
    label: str
    description: str
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    is_mandatory: bool = True


# Catálogo oficial de esquemas para los datasets crudos del proyecto
RAW_DATASETS_SCHEMAS: Dict[str, DatasetSchema] = {
    "dim_equipos.csv": DatasetSchema(
        filename="dim_equipos.csv",
        label="CSV Equipos",
        description="Catálogo maestro de equipos industriales (área de helados)",
        required_columns=[
            "Identificador_Equipo",
            "Codigo_Equipo_Origen",
            "Nombre_Equipo",
            "Tipo_Equipo",
            "Línea",
            "Criticidad",
        ],
        optional_columns=["Área", "Proceso", "Centro", "Año_Instalación", "Estado_Actual", "Fabricante", "Modelo"],
        numeric_columns=["Identificador_Equipo"],
        is_mandatory=True,
    ),
    "dim_sensores.csv": DatasetSchema(
        filename="dim_sensores.csv",
        label="CSV Sensores",
        description="Catálogo de sensores de telemetría instalados por equipo",
        required_columns=[
            "Identificador_Sensor",
            "Identificador_Equipo",
            "Tipo_Sensor",
            "Caracteristica",
            "Unidad_Medida",
        ],
        optional_columns=["Codigo_Equipo_Origen", "Nombre_Equipo", "Nombre_Sensor", "Importancia", "Frecuencia_Medicion", "Proceso_Monitoreado"],
        numeric_columns=["Identificador_Sensor", "Identificador_Equipo"],
        is_mandatory=True,
    ),
    "hechos_fallas.csv": DatasetSchema(
        filename="hechos_fallas.csv",
        label="CSV Mantenimiento y Fallas",
        description="Historial de eventos de falla, paradas y costos de reparación",
        required_columns=[
            "Identificador_Falla",
            "Identificador_Equipo",
            "Fecha_Falla",
            "Tipo_Falla",
            "Duracion_Parada_h",
            "Costo_Estimado_Bs",
        ],
        optional_columns=["Codigo_Equipo_Origen", "Nombre_Equipo", "Severidad", "Causa_Probable"],
        numeric_columns=["Identificador_Equipo", "Duracion_Parada_h", "Costo_Estimado_Bs"],
        date_columns=["Fecha_Falla"],
        is_mandatory=True,
    ),
    "hechos_mediciones_sensores.csv": DatasetSchema(
        filename="hechos_mediciones_sensores.csv",
        label="CSV Mediciones de Sensores",
        description="Registro detallado de telemetría de sensores por turno",
        required_columns=[
            "Identificador_Medicion",
            "Identificador_Equipo",
            "Identificador_Sensor",
            "Fecha_Observacion",
            "Valor_Medicion",
        ],
        optional_columns=["Codigo_Equipo_Origen", "Nombre_Equipo", "Nombre_Sensor", "Tipo_Sensor", "Caracteristica", "Unidad_Medida", "Fecha_Hora_Medicion", "Identificador_Tecnico", "Nombre_Tecnico", "Turno", "Calidad_Medicion"],
        numeric_columns=["Identificador_Equipo", "Identificador_Sensor", "Valor_Medicion"],
        date_columns=["Fecha_Observacion"],
        is_mandatory=True,
    ),
    "observaciones_diarias_equipo.csv": DatasetSchema(
        filename="observaciones_diarias_equipo.csv",
        label="CSV Historial y Observaciones Diarias",
        description="Dataset operativo consolidado diario con variable objetivo para modelado",
        required_columns=[
            "Identificador_Equipo",
            "Fecha_Observacion",
            "Temperatura_Proceso",
            "Vibracion_Equipo",
            "Presion_Sistema",
            "Corriente_Motor",
            "Caudal_Proceso",
            "Nivel_Sistema",
            "Velocidad_Accionamiento",
            "Horas_Operacion",
            "Falla_En_Los_Siguientes_Siete_Dias",
            "Tiene_Horizonte_Completo_Siete_Dias",
            "Conjunto_Temporal",
        ],
        optional_columns=["Codigo_Equipo_Origen", "Nombre_Equipo", "Tipo_Equipo", "Proceso", "Línea", "Criticidad"],
        numeric_columns=[
            "Identificador_Equipo",
            "Temperatura_Proceso",
            "Vibracion_Equipo",
            "Presion_Sistema",
            "Corriente_Motor",
            "Caudal_Proceso",
            "Nivel_Sistema",
            "Velocidad_Accionamiento",
            "Horas_Operacion",
        ],
        date_columns=["Fecha_Observacion"],
        is_mandatory=True,
    ),
    "diccionario_datos.csv": DatasetSchema(
        filename="diccionario_datos.csv",
        label="CSV Diccionario de Datos",
        description="Metadatos descriptivos de variables y entidades del negocio",
        required_columns=["Archivo", "Variable", "Tipo_Dato", "Descripcion"],
        optional_columns=["Ejemplo"],
        is_mandatory=False,
    ),
}


class DataValidator:
    """Validador exhaustivo de datasets crudos antes de la ejecución del pipeline."""

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        self.schemas = RAW_DATASETS_SCHEMAS

    def validate_single_dataframe(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
        source_name: str = "",
    ) -> Dict[str, Any]:
        """Valida un DataFrame en memoria contra su esquema esperado."""
        errors: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # 1. Comprobar si el DataFrame está vacío
        if df.empty:
            errors.append({
                "dataset": schema.filename,
                "column": "General",
                "error_type": "Dataset vacío",
                "affected_rows": 0,
                "description": f"El archivo '{source_name or schema.filename}' no contiene registros.",
            })
            return {
                "status": "warning",
                "filename": schema.filename,
                "label": schema.label,
                "is_mandatory": schema.is_mandatory,
                "row_count": 0,
                "col_count": 0,
                "errors": errors,
                "warnings": warnings,
                "missing_columns": schema.required_columns,
                "unique_equipos": 0,
            }

        row_count = len(df)
        col_count = df.shape[1]

        # 2. Comprobar columnas requeridas
        df_cols_lower = {str(c).strip().lower(): c for c in df.columns}
        missing_cols = []
        for req_col in schema.required_columns:
            if req_col not in df.columns:
                # Comprobar si existe con variación de mayúsculas/minúsculas
                if req_col.lower() in df_cols_lower:
                    warnings.append(f"Columna '{req_col}' encontrada con diferente capitalización: '{df_cols_lower[req_col.lower()]}'")
                else:
                    missing_cols.append(req_col)

        if missing_cols:
            errors.append({
                "dataset": schema.filename,
                "column": ", ".join(missing_cols),
                "error_type": "Columnas requeridas faltantes",
                "affected_rows": row_count,
                "description": f"Faltan las siguientes columnas obligatorias: {', '.join(missing_cols)}",
            })

        # 3. Comprobar tipos numéricos y valores corruptos
        for num_col in schema.numeric_columns:
            if num_col in df.columns:
                # Intentar conversión a numérico
                non_numeric_mask = pd.to_numeric(df[num_col], errors="coerce").isnull() & df[num_col].notnull()
                non_numeric_count = int(non_numeric_mask.sum())
                if non_numeric_count > 0:
                    errors.append({
                        "dataset": schema.filename,
                        "column": num_col,
                        "error_type": "Valores no numéricos",
                        "affected_rows": non_numeric_count,
                        "description": f"La columna numérica '{num_col}' contiene {non_numeric_count:,d} valores no numéricos.",
                    })

        # 4. Comprobar fechas
        for date_col in schema.date_columns:
            if date_col in df.columns:
                invalid_dates_mask = pd.to_datetime(df[date_col], errors="coerce").isnull() & df[date_col].notnull()
                invalid_date_count = int(invalid_dates_mask.sum())
                if invalid_date_count > 0:
                    errors.append({
                        "dataset": schema.filename,
                        "column": date_col,
                        "error_type": "Formato de fecha inválido",
                        "affected_rows": invalid_date_count,
                        "description": f"La columna de fecha '{date_col}' contiene {invalid_date_count:,d} fechas no interpretables.",
                    })

        # 5. Comprobar identificador de equipo si aplica
        unique_equipos = 0
        if "Identificador_Equipo" in df.columns:
            unique_equipos = int(df["Identificador_Equipo"].nunique())

        # Determinar estado
        if errors:
            status = "warning"
        else:
            status = "valid"

        return {
            "status": status,
            "filename": schema.filename,
            "label": schema.label,
            "is_mandatory": schema.is_mandatory,
            "row_count": row_count,
            "col_count": col_count,
            "errors": errors,
            "warnings": warnings,
            "missing_columns": missing_cols,
            "unique_equipos": unique_equipos,
        }

    def validate_file_on_disk(self, filename: str) -> Dict[str, Any]:
        """Valida un archivo guardado en el directorio data/raw/."""
        schema = self.schemas.get(filename)
        if not schema:
            return {
                "status": "warning",
                "filename": filename,
                "label": filename,
                "is_mandatory": False,
                "row_count": 0,
                "col_count": 0,
                "errors": [{"dataset": filename, "column": "General", "error_type": "Esquema desconocido", "affected_rows": 0, "description": f"No hay esquema definido para {filename}"}],
                "warnings": [],
                "missing_columns": [],
                "unique_equipos": 0,
            }

        if not self.raw_data_dir:
            raise ValueError("raw_data_dir no fue especificado al inicializar DataValidator.")

        file_path = self.raw_data_dir / filename
        if not file_path.exists():
            return {
                "status": "missing",
                "filename": filename,
                "label": schema.label,
                "is_mandatory": schema.is_mandatory,
                "row_count": 0,
                "col_count": 0,
                "errors": [{"dataset": filename, "column": "Archivo", "error_type": "Archivo faltante", "affected_rows": 0, "description": f"El archivo '{filename}' no existe en {self.raw_data_dir}"}],
                "warnings": [],
                "missing_columns": schema.required_columns,
                "unique_equipos": 0,
            }

        # Validar extensión .csv
        if file_path.suffix.lower() != ".csv":
            return {
                "status": "warning",
                "filename": filename,
                "label": schema.label,
                "is_mandatory": schema.is_mandatory,
                "row_count": 0,
                "col_count": 0,
                "errors": [{"dataset": filename, "column": "Formato", "error_type": "Extensión inválida", "affected_rows": 0, "description": f"El archivo debe tener extensión .csv (se recibió {file_path.suffix})"}],
                "warnings": [],
                "missing_columns": schema.required_columns,
                "unique_equipos": 0,
            }

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
            except Exception as e:
                return {
                    "status": "warning",
                    "filename": filename,
                    "label": schema.label,
                    "is_mandatory": schema.is_mandatory,
                    "row_count": 0,
                    "col_count": 0,
                    "errors": [{"dataset": filename, "column": "Codificación", "error_type": "Error de lectura CSV", "affected_rows": 0, "description": f"No se pudo leer el archivo CSV: {str(e)}"}],
                    "warnings": [],
                    "missing_columns": schema.required_columns,
                    "unique_equipos": 0,
                }
        except Exception as e:
            return {
                "status": "warning",
                "filename": filename,
                "label": schema.label,
                "is_mandatory": schema.is_mandatory,
                "row_count": 0,
                "col_count": 0,
                "errors": [{"dataset": filename, "column": "General", "error_type": "Error de lectura", "affected_rows": 0, "description": f"Error al abrir CSV: {str(e)}"}],
                "warnings": [],
                "missing_columns": schema.required_columns,
                "unique_equipos": 0,
            }

        return self.validate_single_dataframe(df, schema, source_name=filename)

    def validate_all_mandatory_datasets(self) -> Dict[str, Any]:
        """Valida el conjunto completo de datasets obligatorios en disco."""
        results: Dict[str, Dict[str, Any]] = {}
        all_valid = True
        missing_datasets: List[str] = []
        datasets_with_errors: List[str] = []
        all_errors: List[Dict[str, Any]] = []

        for filename, schema in self.schemas.items():
            if schema.is_mandatory:
                val_res = self.validate_file_on_disk(filename)
                results[filename] = val_res

                if val_res["status"] == "missing":
                    all_valid = False
                    missing_datasets.append(filename)
                    all_errors.extend(val_res["errors"])
                elif val_res["status"] == "warning":
                    all_valid = False
                    datasets_with_errors.append(filename)
                    all_errors.extend(val_res["errors"])

        return {
            "can_execute_pipeline": all_valid,
            "datasets": results,
            "missing_datasets": missing_datasets,
            "datasets_with_errors": datasets_with_errors,
            "all_errors": all_errors,
        }
