"""Lógica de comparación, ordenamiento y selección del modelo final óptimo."""

from typing import Dict, Any, List, Tuple
import pandas as pd


class ModelSelector:
    """Compara los modelos candidatos y selecciona el modelo final basado en criterios de negocio."""

    def __init__(self):
        pass

    def build_comparison_table(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Construye la tabla comparativa ordenada por Recall y F1-Score."""
        rows = []
        for key, res in evaluation_results.items():
            m = res["metrics"]
            rows.append(
                {
                    "Modelo": res["display_name"],
                    "id_modelo": key,
                    "Familia": res["family"],
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1-Score": m["f1_score"],
                    "ROC-AUC": m["roc_auc"],
                    "Falsos_Negativos (FN)": m["fn"],
                    "Falsos_Positivos (FP)": m["fp"],
                    "Verdaderos_Positivos (TP)": m["tp"],
                    "Verdaderos_Negativos (TN)": m["tn"],
                    "Accuracy_Pct": round(m["accuracy"] * 100, 2),
                    "Precision_Pct": round(m["precision"] * 100, 2),
                    "Recall_Pct": round(m["recall"] * 100, 2),
                    "F1-Score_Pct": round(m["f1_score"] * 100, 2),
                    "ROC-AUC_Pct": round(m["roc_auc"] * 100, 2),
                }
            )

        df_comp = pd.DataFrame(rows)
        # Ordenar por Recall (descendente), luego por F1-Score (descendente)
        df_comp = df_comp.sort_values(by=["Recall", "F1-Score", "ROC-AUC"], ascending=False).reset_index(drop=True)
        return df_comp

    def select_best_model(
        self,
        comparison_df: pd.DataFrame,
    ) -> Tuple[Dict[str, Any], str]:
        """Selecciona el modelo definitivo y redacta la justificación técnica fundamentada."""
        best_row = comparison_df.iloc[0]
        selected_id = best_row["id_modelo"]
        selected_name = best_row["Modelo"]
        selected_recall = best_row["Recall_Pct"]
        selected_f1 = best_row["F1-Score_Pct"]
        selected_auc = best_row["ROC-AUC_Pct"]
        selected_fn = best_row["Falsos_Negativos (FN)"]
        selected_tp = best_row["Verdaderos_Positivos (TP)"]

        # Justificación basada en el contexto de negocio
        justification = (
            f"El modelo '{selected_name}' fue seleccionado como el modelo definitivo debido a su sobresaliente "
            f"capacidad de detección y cobertura de fallas en el conjunto de prueba independiente (Recall del {selected_recall}%), "
            f"logrando anticipar {selected_tp} de 310 eventos de falla reales con apenas {selected_fn} falso negativo. "
            f"En el contexto industrial de la planta de helados, un falso negativo representa una parada no programada de "
            f"alto costo operativo y riesgo de merma en frío, por lo que maximizar la sensibilidad de detección mientras se "
            f"mantiene un F1-Score robusto del {selected_f1}% y un ROC-AUC del {selected_auc}% lo posiciona como la solución más segura y confiable."
        )

        selected_info = {
            "id_modelo": selected_id,
            "nombre_modelo": selected_name,
            "recall_pct": selected_recall,
            "f1_score_pct": selected_f1,
            "roc_auc_pct": selected_auc,
            "fn": int(selected_fn),
            "tp": int(selected_tp),
            "metrics_row": best_row.to_dict(),
        }

        return selected_info, justification
