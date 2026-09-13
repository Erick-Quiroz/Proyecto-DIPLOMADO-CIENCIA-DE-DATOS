"""Analizador de series temporales de fallas y evolución en ventana previa."""

import logging
from typing import Dict
import pandas as pd
from services.eda.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger("eda.temporal")


class TemporalAnalyzer(BaseAnalyzer):
    """Analiza la dinámica temporal de eventos de falla y la trayectoria precursora de fallas."""

    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        result = AnalysisResult()
        fallas = datasets.get("hechos_fallas.csv")
        equipos = datasets.get("dim_equipos.csv")
        obs = datasets.get("observaciones_diarias_equipo.csv")

        if fallas is None:
            logger.warning("hechos_fallas.csv no encontrado.")
            return result

        df_fallas = fallas.copy()
        result.findings.append(
            f"El registro histórico contiene {len(df_fallas):,} eventos de falla."
        )

        # 1. Fallas por mes
        if "Fecha_Falla" in df_fallas.columns:
            df_fallas["Fecha_Falla_dt"] = pd.to_datetime(
                df_fallas["Fecha_Falla"], errors="coerce"
            )
            df_fallas["Mes"] = df_fallas["Fecha_Falla_dt"].dt.to_period("M").astype(str)

            fallas_mes = (
                df_fallas.groupby("Mes")
                .size()
                .reset_index(name="Cantidad_Fallas")
            )
            result.tables["12_fallas_por_mes.csv"] = fallas_mes
            result.metadata["fallas_mes"] = fallas_mes

        # 2. Fallas por equipo
        if "Identificador_Equipo" in df_fallas.columns:
            fallas_equipo = (
                df_fallas.groupby("Identificador_Equipo")
                .size()
                .reset_index(name="Cantidad_Fallas")
                .sort_values("Cantidad_Fallas", ascending=False)
            )
            result.tables["13_fallas_por_equipo.csv"] = fallas_equipo
            result.metadata["fallas_equipo"] = fallas_equipo

        # 3. Fallas por tipo de equipo y criticidad
        if equipos is not None and "Identificador_Equipo" in df_fallas.columns and "Identificador_Equipo" in equipos.columns:
            merge_cols = [
                c for c in ["Identificador_Equipo", "Nombre_Equipo", "Tipo_Equipo", "Criticidad", "Proceso"]
                if c in equipos.columns
            ]
            fallas_info = df_fallas.merge(
                equipos[merge_cols],
                on="Identificador_Equipo",
                how="left",
            )

            if "Tipo_Equipo" in fallas_info.columns:
                por_tipo = (
                    fallas_info.groupby("Tipo_Equipo")
                    .size()
                    .reset_index(name="Cantidad_Fallas")
                    .sort_values("Cantidad_Fallas", ascending=False)
                )
                result.tables["14_fallas_por_tipo_equipo.csv"] = por_tipo

            if "Criticidad" in fallas_info.columns:
                por_criticidad = (
                    fallas_info.groupby("Criticidad")
                    .size()
                    .reset_index(name="Cantidad_Fallas")
                    .sort_values("Cantidad_Fallas", ascending=False)
                )
                result.tables["15_fallas_por_criticidad.csv"] = por_criticidad

            if "Proceso" in fallas_info.columns:
                fallas_proceso = (
                    fallas_info.groupby("Proceso", dropna=False)
                    .size()
                    .reset_index(name="Cantidad_Fallas")
                    .sort_values("Cantidad_Fallas", ascending=False)
                )
                fallas_proceso["Porcentaje"] = (
                    fallas_proceso["Cantidad_Fallas"] / len(fallas_info) * 100
                ).round(2)
                result.tables["20_fallas_por_proceso.csv"] = fallas_proceso
                result.metadata["fallas_proceso"] = fallas_proceso

                if not fallas_proceso.empty:
                    top_proc = fallas_proceso.iloc[0]
                    result.findings.append(
                        f"El proceso con mayor cantidad de eventos de falla es {top_proc['Proceso']}, "
                        f"con {int(top_proc['Cantidad_Fallas']):,} eventos ({top_proc['Porcentaje']:.2f}%)."
                    )

        # 4. Evolución previa a la falla (ventana de 21 días)
        if obs is not None and "Identificador_Equipo" in obs.columns and "Fecha_Observacion" in obs.columns and "Fecha_Falla" in df_fallas.columns:
            obs_temp = obs.copy()
            obs_temp["_Equipo_Clave"] = obs_temp["Identificador_Equipo"].astype(str).str.strip()
            df_fallas["_Equipo_Clave"] = df_fallas["Identificador_Equipo"].astype(str).str.strip()

            obs_temp["Fecha_Observacion_dt"] = pd.to_datetime(
                obs_temp["Fecha_Observacion"], errors="coerce"
            ).dt.normalize()
            df_fallas["Fecha_Falla_dt"] = pd.to_datetime(
                df_fallas["Fecha_Falla"], errors="coerce"
            ).dt.normalize()

            fallas_validas = (
                df_fallas.dropna(subset=["_Equipo_Clave", "Fecha_Falla_dt"])
                .drop_duplicates(subset=["_Equipo_Clave", "Fecha_Falla_dt"])
                .copy()
            )

            filas_previas = []
            window_days = self.config.pre_failure_window_days

            for _, evento in fallas_validas.iterrows():
                eq = evento["_Equipo_Clave"]
                ff = evento["Fecha_Falla_dt"]

                grupo = obs_temp[
                    (obs_temp["_Equipo_Clave"] == eq)
                    & (obs_temp["Fecha_Observacion_dt"] < ff)
                    & (obs_temp["Fecha_Observacion_dt"] >= ff - pd.Timedelta(days=window_days))
                ].copy()

                if grupo.empty:
                    continue

                grupo["Dias_Antes_Falla"] = (ff - grupo["Fecha_Observacion_dt"]).dt.days
                filas_previas.append(grupo)

            if filas_previas:
                prev_df = pd.concat(filas_previas, ignore_index=True)
                op_vars = [c for c in self.config.operational_variables if c in prev_df.columns]

                for v in op_vars:
                    prev_df[v] = pd.to_numeric(prev_df[v], errors="coerce")

                resumen_temporal = (
                    prev_df.groupby("Dias_Antes_Falla")[op_vars]
                    .mean()
                    .reset_index()
                    .sort_values("Dias_Antes_Falla")
                )
                result.tables["21_evolucion_previa_falla.csv"] = resumen_temporal
                result.metadata["resumen_temporal_previo"] = resumen_temporal

        return result
