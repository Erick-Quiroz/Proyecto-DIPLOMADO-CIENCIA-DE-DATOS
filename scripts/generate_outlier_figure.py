"""Script para generar una figura PNG de alta resolución sobre el Tratamiento de Valores Atípicos (Outliers)."""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Definir rutas
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "observaciones_diarias_equipo.csv"
OUT_FIG_1 = BASE_DIR / "figuras_preparacion" / "auditoria_outliers.png"
OUT_FIG_2 = BASE_DIR / "reports" / "figures" / "preparacion" / "auditoria_outliers.png"
ARTIFACT_DIR = Path(r"C:\Users\Erick Quiroz\.gemini\antigravity-ide\brain\72553f7d-3043-41f3-9e25-ced9d77b36b5")

OUT_FIG_1.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG_2.parent.mkdir(parents=True, exist_ok=True)

# Cargar datos
df = pd.read_csv(RAW_DATA_PATH)

variables = [
    ("Temperatura_Proceso", "Temperatura Proceso (°C)", "#2563eb"),
    ("Vibracion_Equipo", "Vibración Equipo (mm/s)", "#d97706"),
    ("Presion_Sistema", "Presión Sistema (bar)", "#dc2626"),
    ("Corriente_Motor", "Corriente Motor (A)", "#7c3aed"),
    ("Caudal_Proceso", "Caudal Proceso (m³/h)", "#059669"),
    ("Nivel_Sistema", "Nivel Sistema (%)", "#0891b2"),
    ("Velocidad_Accionamiento", "Velocidad Accionamiento (RPM)", "#4f46e5"),
    ("Horas_Operacion", "Horas de Operación (h)", "#475569"),
]

# Calcular métricas IQR y conteos
stats_list = []
for col, label, color in variables:
    series = df[col].dropna()
    q1 = float(series.quantile(0.25))
    med = float(series.median())
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    min_val = float(series.min())
    max_val = float(series.max())
    outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
    pct = (outliers / len(series)) * 100

    stats_list.append({
        "col": col,
        "label": label,
        "color": color,
        "min": min_val,
        "q1": q1,
        "med": med,
        "q3": q3,
        "max": max_val,
        "lower": lower_bound,
        "upper": upper_bound,
        "outliers": outliers,
        "pct": pct,
        "series": series
    })

# Configurar estilo general
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

fig = plt.figure(figsize=(18, 13), dpi=220)
gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.42, wspace=0.25)

# Título principal
fig.suptitle(
    "Auditoría y Tratamiento de Valores Atípicos (Método IQR - Rango Intercuartílico)\nDecisión Técnica de Ingeniería: Conservar (Condición Operativa Real de Falla)",
    fontsize=16,
    fontweight="bold",
    color="#0f172a",
    y=0.98
)

# Renderizar los 8 subplots de boxplots
for i, stat in enumerate(stats_list):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    
    # Boxplot horizontal
    bp = ax.boxplot(
        stat["series"],
        tick_labels=[""],
        orientation="horizontal",
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=3.5, alpha=0.6, markeredgecolor='none'),
        medianprops=dict(color='#ffffff', linewidth=2.2),
        boxprops=dict(facecolor=stat["color"], color='#1e293b', alpha=0.85, linewidth=1.2),
        whiskerprops=dict(color='#334155', linewidth=1.2, linestyle='-'),
        capprops=dict(color='#334155', linewidth=1.5)
    )

    # Líneas de umbral IQR
    if stat["lower"] >= stat["min"]:
        ax.axvline(stat["lower"], color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.8)
    if stat["upper"] <= stat["max"]:
        ax.axvline(stat["upper"], color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.8)

    # Decoración y títulos de subplots
    ax.set_title(stat["label"], fontsize=11, fontweight="bold", color="#1e293b", pad=8)
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="x")
    ax.tick_params(axis="x", labelsize=9)
    
    # Info badge
    if stat["outliers"] > 0:
        badge_text = f"Atípicos: {stat['outliers']:,} ({stat['pct']:.2f}%)\nLím. Sup: {stat['upper']:.2f}"
        ax.text(
            0.96, 0.15,
            badge_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#f87171", alpha=0.9)
        )
    else:
        badge_text = f"Atípicos: 0 (0.00%)\nSin anomalías Tukey"
        ax.text(
            0.96, 0.15,
            badge_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#15803d",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#86efac", alpha=0.9)
        )

# Renderizar Tabla Resumen en la fila 3 (ocupa las 4 columnas)
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('off')

table_data = [
    [
        s["col"],
        f"{s['min']:,.3f}",
        f"{s['q1']:,.3f}",
        f"{s['med']:,.3f}",
        f"{s['q3']:,.3f}",
        f"{s['max']:,.3f}",
        f"{s['lower']:,.3f}",
        f"{s['upper']:,.3f}",
        f"{s['outliers']:,}",
        f"{s['pct']:.2f} %",
        "Conservar (Señal de Falla)"
    ]
    for s in stats_list
]

columns = [
    "Variable", "Mínimo", "Q1 (25%)", "Mediana", "Q3 (75%)", "Máximo",
    "Lím. Inf. IQR", "Lím. Sup. IQR", "Atípicos", "% Atíp.", "Decisión Técnica"
]

# Definir anchos relativos específicos para que no se corte ningún texto
col_widths = [0.18, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.07, 0.07, 0.16]

table = ax_table.table(
    cellText=table_data,
    colLabels=columns,
    colWidths=col_widths,
    cellLoc='center',
    loc='center',
    bbox=[-0.02, 0.02, 1.04, 0.94]
)

table.auto_set_font_size(False)
table.set_fontsize(9)

# Estilo de tabla
for (row, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor('#94a3b8')
    cell.set_linewidth(0.8)
    if row == 0:
        cell.set_facecolor('#0f172a')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_height(0.13)
    else:
        # Colores alternados
        if row % 2 == 0:
            cell.set_facecolor('#f8fafc')
        else:
            cell.set_facecolor('#ffffff')
        
        # Columna de nombres a la izquierda
        if col_idx == 0:
            cell.set_text_props(ha='left', fontfamily='monospace', fontsize=8.5)
        
        # Columna decisión
        if col_idx == 10:
            cell.set_facecolor('#ecfdf5')
            cell.set_text_props(color='#047857', fontweight='bold', fontsize=8.5)
        # Columna atípicos
        if col_idx in (8, 9):
            if table_data[row-1][8] != "0":
                cell.set_facecolor('#fef2f2')
                cell.set_text_props(color='#b91c1c', fontweight='bold')

# Guardar figura
plt.savefig(OUT_FIG_1, dpi=220, bbox_inches='tight')
plt.savefig(OUT_FIG_2, dpi=220, bbox_inches='tight')

# Copiar a artifact dir si existe
if ARTIFACT_DIR.exists():
    artifact_fig = ARTIFACT_DIR / "auditoria_outliers.png"
    plt.savefig(artifact_fig, dpi=220, bbox_inches='tight')

plt.close(fig)
print(f"Figura generada con éxito:\n - {OUT_FIG_1}\n - {OUT_FIG_2}")

