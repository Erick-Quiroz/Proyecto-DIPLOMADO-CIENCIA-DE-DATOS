"""Genera una nueva tabla y figura comparativa incluyendo el Modelo Baseline (Dummy Classifier).

Este script NO modifica datasets existentes, hiperparámetros ni resultados previos de los modelos entrenados.
Lee las métricas existentes y calcula el baseline para generar artefactos nuevos independientes:
- results/evaluation/tabla_comparativa_con_baseline.csv
- results/evaluation/tabla_comparativa_con_baseline.png
- reports/figures/evaluacion/tabla_comparativa_con_baseline.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuración de rutas relativas al proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
y_train_path = PROJECT_ROOT / "data" / "processed" / "y_train.csv"
y_test_path = PROJECT_ROOT / "data" / "processed" / "y_test.csv"
existing_csv = PROJECT_ROOT / "results" / "evaluation" / "tabla_comparacion_modelos.csv"

output_csv = PROJECT_ROOT / "results" / "evaluation" / "tabla_comparativa_con_baseline.csv"
output_png = PROJECT_ROOT / "results" / "evaluation" / "tabla_comparativa_con_baseline.png"
reports_png = PROJECT_ROOT / "reports" / "figures" / "evaluacion" / "tabla_comparativa_con_baseline.png"

# Crear directorios de destino si no existen
output_png.parent.mkdir(parents=True, exist_ok=True)
reports_png.parent.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos de etiquetas para computar el Baseline real
if y_train_path.exists() and y_test_path.exists():
    y_train = pd.read_csv(y_train_path, encoding="utf-8-sig").squeeze().astype(int)
    y_test = pd.read_csv(y_test_path, encoding="utf-8-sig").squeeze().astype(int)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([0] * len(y_train), y_train)
    y_pred_dummy = dummy.predict([0] * len(y_test))
    y_prob_dummy = dummy.predict_proba([0] * len(y_test))[:, 1]

    baseline_row = {
        "Modelo": "Baseline (Clase Mayoritaria)",
        "Accuracy": accuracy_score(y_test, y_pred_dummy),
        "Precision": precision_score(y_test, y_pred_dummy, zero_division=0),
        "Recall": recall_score(y_test, y_pred_dummy, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_dummy, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob_dummy),
    }
else:
    # Valores por defecto calculados previamente en caso de aislamiento
    baseline_row = {
        "Modelo": "Baseline (Clase Mayoritaria)",
        "Accuracy": 0.872742,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1-Score": 0.0,
        "ROC-AUC": 0.5,
    }

# 2. Cargar tabla existente de modelos y unir el baseline al inicio
df_models = pd.read_csv(existing_csv)
df_baseline = pd.DataFrame([baseline_row])
df_combined = pd.concat([df_baseline, df_models], ignore_index=True)

# Guardar la nueva tabla comparativa CSV (sin sobreescribir la original)
df_combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"[OK] Nueva tabla guardada en: {output_csv}")

# 3. Generación de la Figura Gráfica Comparativa (Tabla + Barras)
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 8), dpi=300, gridspec_kw={"height_ratios": [1.1, 1.5]}
)

# Renderizado de la tabla superior
ax1.axis("off")
columns = ["Modelo", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"]
table_data = []
for _, row in df_combined.iterrows():
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
table.set_fontsize(9.5)
table.scale(1.0, 1.8)

# Estilizado de celdas
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(color="white", weight="bold")
        cell.set_facecolor("#1f3a52")
    elif r == 1:
        # Resaltar la fila de Baseline con tono gris neutro
        cell.set_facecolor("#e2e8f0")
        cell.set_text_props(style="italic")
    else:
        cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")

ax1.set_title(
    "TABLA COMPARATIVA: MODELOS ENTRENADOS VS. BASELINE (Conjunto Test)",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

# Gráfico de barras agrupadas comparativo
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics))
n_models = len(df_combined)
width = 0.18

colors = {
    "Baseline (Clase Mayoritaria)": "#64748b",
    "Regresión Logística": "#1f77b4",
    "Random Forest": "#2ca02c",
    "XGBoost": "#ff7f0e",
}

for i, (_, row) in enumerate(df_combined.iterrows()):
    model_name = row["Modelo"]
    values = [row[m] * 100 for m in metrics]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax2.bar(
        x + offset,
        values,
        width,
        label=model_name,
        color=colors.get(model_name, "#888888"),
        alpha=0.92,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                weight="bold",
            )

ax2.set_ylabel("Rendimiento (%)", fontsize=10, fontweight="bold")
ax2.set_title(
    "Comparativa de Rendimiento: Ganancia de Valor sobre el Baseline",
    fontsize=11,
    fontweight="bold",
)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10, fontweight="bold")
ax2.set_ylim(0, 115)
ax2.legend(loc="upper left", frameon=True, fontsize=8.5)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(output_png, bbox_inches="tight")
fig.savefig(reports_png, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Nueva figura generada en:\n - {output_png}\n - {reports_png}")
