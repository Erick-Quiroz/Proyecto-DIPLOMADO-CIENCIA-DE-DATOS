"""Analizador estadístico: estadísticas descriptivas, atípicos IQR y correlaciones."""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from services.eda.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger("eda.statistical")


class StatisticalAnalyzer(BaseAnalyzer):
    """Calcula métricas descriptivas, detección de outliers IQR y matrices de correlación."""

    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        result = AnalysisResult()
        if not datasets:
            return result

        # 1. Estadística descriptiva global
        estadisticas = []
        for nombre, df in datasets.items():
            num = df.select_dtypes(include=np.number)
            if not num.empty:
                x = (
                    num.describe()
                    .T
                    .reset_index()
                    .rename(columns={"index": "Variable"})
                )
                x.insert(0, "Archivo", nombre)
                estadisticas.append(x)

        if estadisticas:
            df_estadisticas = pd.concat(estadisticas, ignore_index=True)
            result.tables["07_estadistica_descriptiva.csv"] = df_estadisticas

        # 2. Atípicos IQR y correlación en observaciones_diarias_equipo.csv
        obs = datasets.get("observaciones_diarias_equipo.csv")
        if obs is not None:
            op_vars = [c for c in self.config.operational_variables if c in obs.columns]

            # Atípicos IQR
            outliers_rows = []
            for col in op_vars:
                s = pd.to_numeric(obs[col], errors="coerce").dropna()
                if len(s) == 0:
                    continue

                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lim_inf = q1 - self.config.iqr_multiplier * iqr
                lim_sup = q3 + self.config.iqr_multiplier * iqr
                mask = (s < lim_inf) | (s > lim_sup)

                outliers_rows.append({
                    "Variable": col,
                    "Q1": q1,
                    "Mediana": s.median(),
                    "Q3": q3,
                    "Limite_Inferior": lim_inf,
                    "Limite_Superior": lim_sup,
                    "Outliers": int(mask.sum()),
                    "Porcentaje_Outliers": round(mask.mean() * 100, 2),
                })

            df_outliers = pd.DataFrame(outliers_rows)
            if not df_outliers.empty:
                df_outliers = df_outliers.sort_values(
                    "Porcentaje_Outliers", ascending=False
                ).reset_index(drop=True)
                result.tables["09_outliers_IQR.csv"] = df_outliers

                top_out = df_outliers.iloc[0]
                result.findings.append(
                    f"La mayor proporción de posibles valores atípicos según el criterio IQR "
                    f"corresponde a {top_out['Variable']} ({top_out['Porcentaje_Outliers']:.2f}%)."
                )
            else:
                result.findings.append(
                    "No se identificaron variables con posibles valores atípicos según el criterio IQR."
                )

            # Matriz de correlación
            if op_vars:
                corr = obs[op_vars].corr()
                result.tables["10_matriz_correlacion.csv"] = corr
                result.metadata["correlation_matrix"] = corr

        return result
