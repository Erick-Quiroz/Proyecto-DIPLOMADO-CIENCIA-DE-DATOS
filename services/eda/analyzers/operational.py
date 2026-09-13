"""Analizador de validación física por proceso y cobertura de sensores."""

import logging
from typing import Dict
import pandas as pd
from services.eda.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger("eda.operational")


class OperationalAnalyzer(BaseAnalyzer):
    """Valida los rangos físicos de operación por proceso y la cobertura de los sensores."""

    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        result = AnalysisResult()
        obs = datasets.get("observaciones_diarias_equipo.csv")

        # 1. Validación física de rangos por proceso
        if obs is not None and "Proceso" in obs.columns:
            op_vars = [c for c in self.config.operational_variables if c in obs.columns]
            resumen_proceso = []

            for proceso, grupo in obs.groupby("Proceso", dropna=False):
                for variable in op_vars:
                    serie = pd.to_numeric(grupo[variable], errors="coerce").dropna()
                    if len(serie) == 0:
                        continue

                    resumen_proceso.append({
                        "Proceso": proceso,
                        "Variable": variable,
                        "N": len(serie),
                        "Minimo": serie.min(),
                        "P25": serie.quantile(0.25),
                        "Mediana": serie.median(),
                        "P75": serie.quantile(0.75),
                        "Maximo": serie.max(),
                    })

            if resumen_proceso:
                df_rangos = pd.DataFrame(resumen_proceso)
                result.tables["18_rangos_por_proceso.csv"] = df_rangos

            if "Identificador_Equipo" in obs.columns:
                df_eq_proc = (
                    obs.groupby("Proceso")["Identificador_Equipo"]
                    .nunique()
                    .reset_index(name="Cantidad_Equipos")
                )
                result.tables["19_equipos_por_proceso.csv"] = df_eq_proc

        # 2. Cobertura de sensores
        mediciones = datasets.get("hechos_mediciones_sensores.csv")
        if mediciones is not None:
            cols_cobertura = [
                c for c in [
                    "Identificador_Sensor",
                    "Tipo_Sensor",
                    "Nombre_Sensor",
                    "Identificador_Equipo",
                ]
                if c in mediciones.columns
            ]
            if cols_cobertura:
                cobertura = (
                    mediciones.groupby(cols_cobertura, dropna=False)
                    .size()
                    .reset_index(name="Cantidad_Mediciones")
                )
                result.tables["16_cobertura_sensores.csv"] = cobertura

        return result
