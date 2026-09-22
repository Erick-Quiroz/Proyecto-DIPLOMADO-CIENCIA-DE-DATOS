"""Genera la figura gráfica (.png) de la tabla comparativa de métricas."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
input_csv = PROJECT_ROOT / "results" / "evaluation" / "tabla_comparacion_modelos.csv"
output_png = PROJECT_ROOT / "results" / "evaluation" / "tabla_comparacion_modelos.png"
reports_png = PROJECT_ROOT / "reports" / "figures" / "evaluacion" / "tabla_comparacion_modelos.png"

reports_png.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=300, gridspec_kw={"height_ratios": [1, 1.4]})

# 1. Tabla estilizada renderizada
ax1.axis("off")
columns = ["Modelo", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"]
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row["Modelo"],
        f"{row['Accuracy']*100:.2f} %",
        f"{row['Precision']*100:.2f} %",
        f"{row['Recall']*100:.2f} %",
        f"{row['F1-Score']*100:.2f} %",
        f"{row['ROC-AUC']*100:.2f} %",
    ])

table = ax1.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(color="white", weight="bold")
        cell.set_facecolor("#1f3a52")
    else:
        cell.set_facecolor("#f1f5f9" if r % 2 == 1 else "#ffffff")

ax1.set_title("TABLA COMPARATIVA DE MÉTRICAS (Conjunto de Prueba - Test)", fontsize=12, fontweight="bold", pad=15)

# 2. Gráfico de barras agrupadas comparativo
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.25

colors = {"Random Forest": "#2ca02c", "Regresión Logística": "#1f77b4", "XGBoost": "#ff7f0e"}
for i, (_, row) in enumerate(df.iterrows()):
    model_name = row["Modelo"]
    values = [row[m] * 100 for m in metrics]
    bars = ax2.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model_name,
        color=colors.get(model_name, "#555555"),
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

ax2.set_ylabel("Porcentaje (%)", fontsize=10, fontweight="bold")
ax2.set_title("Comparativa Gráfica de Rendimiento por Métrica", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10, fontweight="bold")
ax2.set_ylim(60, 105)
ax2.legend(loc="lower right", frameon=True)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(output_png, bbox_inches="tight")
fig.savefig(reports_png, bbox_inches="tight")
plt.close(fig)

print(f"Figura generada en:\n - {output_png}\n - {reports_png}")
