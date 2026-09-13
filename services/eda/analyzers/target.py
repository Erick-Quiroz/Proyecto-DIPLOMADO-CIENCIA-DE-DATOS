"""Analizador de la variable objetivo y validación del horizonte de 7 días."""

import logging
from typing import Dict
import pandas as pd
from services.eda.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger("eda.target")


class TargetAnalyzer(BaseAnalyzer):
    """Analiza la variable objetivo, su balance de clases y la consistencia metodológica del horizonte."""

    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        result = AnalysisResult()
        obs = datasets.get("observaciones_diarias_equipo.csv")
        target_col = self.config.target_column

        if obs is None:
            logger.warning("observaciones_diarias_equipo.csv no encontrado.")
            return result

        # 1. Resumen de distribución del target
        if target_col in obs.columns:
            resumen_target = pd.DataFrame({
                "Cantidad": obs[target_col].value_counts(dropna=False),
                "Porcentaje": (
                    obs[target_col].value_counts(normalize=True, dropna=False) * 100
                ).round(2),
            })
            result.tables["08_variable_objetivo.csv"] = resumen_target

            n0 = int(obs[target_col].eq(0).sum())
            n1 = int(obs[target_col].eq(1).sum())
            ns = int(obs[target_col].isna().sum())

            result.metadata["target_counts"] = {
                "No falla (0)": n0,
                "Falla (1)": n1,
                "Sin etiqueta": ns,
            }

            # Hallazgos de estructura
            eq_col = "Identificador_Equipo" if "Identificador_Equipo" in obs.columns else None
            n_equipos = obs[eq_col].nunique() if eq_col else 0
            fecha_col = "Fecha_Observacion" if "Fecha_Observacion" in obs.columns else None

            result.findings.append(
                f"El conjunto de datos estructurado bajo la unidad de análisis Equipo-Día contiene "
                f"{len(obs):,} observaciones y {obs.shape[1]:,} variables."
            )
            if eq_col:
                result.findings.append(f"Se identificaron {n_equipos:,} equipos.")

            if fecha_col:
                s_fecha = pd.to_datetime(obs[fecha_col], errors="coerce")
                if s_fecha.notna().any():
                    result.findings.append(
                        f"El período de observación comprende desde {s_fecha.min().date()} "
                        f"hasta {s_fecha.max().date()}."
                    )

            result.findings.append(
                f"La variable objetivo presenta {n0:,} observaciones con valor 0, "
                f"{n1:,} con valor 1 y {ns:,} sin etiqueta."
            )

        # 2. Comparación de variables operativas según ocurrencia de falla
        if target_col in obs.columns:
            etiquetados = obs[obs[target_col].isin([0, 1])].copy()
            op_vars = [c for c in self.config.operational_variables if c in etiquetados.columns]

            comparacion = []
            for col in op_vars:
                temp = etiquetados[[target_col, col]].copy()
                temp[col] = pd.to_numeric(temp[col], errors="coerce")
                temp = temp.dropna()

                resumen = (
                    temp.groupby(target_col)[col]
                    .agg(
                        Cantidad="count",
                        Media="mean",
                        Mediana="median",
                        Desviacion="std",
                        Minimo="min",
                        Maximo="max",
                    )
                    .reset_index()
                )
                resumen.insert(0, "Variable", col)
                comparacion.append(resumen)

            if comparacion:
                df_comp = pd.concat(comparacion, ignore_index=True)
                result.tables["11_comparacion_por_falla.csv"] = df_comp

        return result
