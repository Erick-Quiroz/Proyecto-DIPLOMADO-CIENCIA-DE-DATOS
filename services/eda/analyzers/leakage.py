"""Auditor de fuga de información y matriz de decisión de variables predictoras."""

import logging
from typing import Dict
import pandas as pd
from services.eda.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger("eda.leakage")


class DataLeakageAuditor(BaseAnalyzer):
    """Evalúa los riesgos de fuga de información y categoriza las variables candidatas para ML."""

    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        result = AnalysisResult()
        obs = datasets.get("observaciones_diarias_equipo.csv")

        if obs is None:
            return result

        columnas = list(obs.columns)
        target = self.config.target_column
        risk_keywords = self.config.leakage_risk_keywords

        reglas = []
        for c in columnas:
            lc = c.lower()

            if c == target:
                decision = "EXCLUIR"
                motivo = "Variable objetivo del modelo"
            elif any(k in lc for k in risk_keywords):
                decision = "EXCLUIR"
                motivo = "Puede contener información del evento o una variable derivada posterior"
            elif c in [
                "Tiene_Horizonte_Completo_Siete_Dias",
                "Conjunto_Temporal",
            ]:
                decision = "CONTROL"
                motivo = "Variable metodológica; no debe utilizarse como predictor"
            elif c in [
                "Identificador_Equipo",
                "Codigo_Equipo_Origen",
                "Nombre_Equipo",
            ]:
                decision = "REVISAR"
                motivo = "Identificación o categoría; requiere codificación o exclusión"
            elif c == "Fecha_Observacion":
                decision = "REVISAR"
                motivo = "Puede transformarse en variables temporales sin usar información futura"
            else:
                decision = "CANDIDATA"
                motivo = "Información disponible al cierre de la observación"

            reglas.append({
                "Variable": c,
                "Decisión": decision,
                "Justificación": motivo,
            })

        df_fuga = pd.DataFrame(reglas)
        result.tables["22_matriz_fuga_informacion.csv"] = df_fuga

        return result
