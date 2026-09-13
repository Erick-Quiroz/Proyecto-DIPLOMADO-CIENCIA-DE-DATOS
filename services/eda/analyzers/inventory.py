"""Analizador de inventario general, tipos de datos, calidad y frecuencias."""

import logging
from typing import Dict, List
import pandas as pd
from services.eda.analyzers.base import BaseAnalyzer, AnalysisResult

logger = logging.getLogger("eda.inventory")


class InventoryAnalyzer(BaseAnalyzer):
    """Inspecciona la integridad básica, esquemas, tipos, nulos, duplicados y cardinalidad."""

    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        result = AnalysisResult()
        if not datasets:
            return result

        # 1. Inventario general
        inventario_rows = []
        for nombre, df in datasets.items():
            inventario_rows.append({
                "Archivo": nombre,
                "Registros": len(df),
                "Variables": df.shape[1],
            })
        df_inventario = pd.DataFrame(inventario_rows).sort_values("Archivo").reset_index(drop=True)
        result.tables["01_inventario.csv"] = df_inventario

        # 2. Variables, tipos y calidad
        estructura_rows = []
        for nombre, df in datasets.items():
            for col in df.columns:
                nulos = int(df[col].isna().sum())
                pct_nulos = round(df[col].isna().mean() * 100, 2)
                uniques = int(df[col].nunique(dropna=True))
                estructura_rows.append({
                    "Archivo": nombre,
                    "Variable": col,
                    "Tipo_Dato": str(df[col].dtype),
                    "Nulos": nulos,
                    "Porcentaje_Nulos": pct_nulos,
                    "Valores_Unicos": uniques,
                })
        df_estructura = pd.DataFrame(estructura_rows)
        result.tables["02_variables_tipos_calidad.csv"] = df_estructura

        # 3. Valores faltantes
        df_faltantes = df_estructura[df_estructura["Nulos"] > 0].sort_values(
            "Porcentaje_Nulos", ascending=False
        ).reset_index(drop=True)
        result.tables["03_valores_faltantes.csv"] = df_faltantes

        # 4. Duplicados completos
        duplicados_rows = []
        for nombre, df in datasets.items():
            dups = int(df.duplicated().sum())
            pct_dups = round(df.duplicated().mean() * 100, 2) if len(df) else 0.0
            duplicados_rows.append({
                "Archivo": nombre,
                "Registros": len(df),
                "Duplicados_Completos": dups,
                "Porcentaje_Duplicados": pct_dups,
            })
        df_duplicados = pd.DataFrame(duplicados_rows)
        result.tables["04_duplicados.csv"] = df_duplicados

        # 5. Revisión de fechas
        fechas_rows = []
        for nombre, df in datasets.items():
            for col in df.columns:
                if "fecha" in col.lower() or "hora" in col.lower():
                    s = pd.to_datetime(df[col], errors="coerce")
                    fechas_rows.append({
                        "Archivo": nombre,
                        "Variable": col,
                        "Convertibles": int(s.notna().sum()),
                        "No_Convertibles": int(s.isna().sum()),
                        "Minimo": s.min(),
                        "Maximo": s.max(),
                    })
        df_fechas = pd.DataFrame(fechas_rows)
        result.tables["05_revision_fechas.csv"] = df_fechas

        # 6. Frecuencias categóricas
        cat_rows = []
        for nombre, df in datasets.items():
            for col in df.select_dtypes(include="object").columns:
                if df[col].nunique(dropna=True) <= self.config.max_categorical_cardinality:
                    for val, count in df[col].value_counts(dropna=False).items():
                        cat_rows.append({
                            "Archivo": nombre,
                            "Variable": col,
                            "Valor": val,
                            "Frecuencia": int(count),
                        })
        df_cat = pd.DataFrame(cat_rows)
        result.tables["06_frecuencias_categoricas.csv"] = df_cat

        # Síntesis de hallazgos
        if not df_faltantes.empty:
            top_null = df_faltantes.iloc[0]
            result.findings.append(
                f"La variable con mayor proporción de valores faltantes es "
                f"{top_null['Variable']} con {top_null['Porcentaje_Nulos']:.2f}%."
            )
        else:
            result.findings.append(
                "No se identificaron valores faltantes en las variables revisadas."
            )

        total_dups = int(df_duplicados["Duplicados_Completos"].sum())
        result.findings.append(
            f"Se detectaron {total_dups:,} registros completamente duplicados."
        )

        return result
