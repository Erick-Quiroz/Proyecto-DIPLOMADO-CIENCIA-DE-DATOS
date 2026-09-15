"""Evaluador de métricas, inferencia y generación de curvas de desempeño."""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

from services.evaluation.config import EvaluationConfig


class ModelEvaluator:
    """Calcula métricas de clasificación, genera matrices de confusión y curvas ROC."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate_model_on_split(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """Calcula todas las métricas de clasificación binaria para un conjunto de datos."""
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_proba)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "specificity": specificity,
            "confusion_matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    def generate_stability_analysis(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[Dict[str, Any]]:
        """Compara métricas entre Train, Valid y Test para detectar overfitting o underfitting."""
        stability_rows = []
        splits = [
            ("Train", X_train, y_train),
            ("Valid", X_valid, y_valid),
            ("Test", X_test, y_test),
        ]

        for key, model_data in models.items():
            estimator = model_data["estimator"]
            display_name = model_data["display_name"]

            for split_name, X_split, y_split in splits:
                metrics = self.evaluate_model_on_split(estimator, X_split, y_split)
                stability_rows.append(
                    {
                        "id_modelo": key,
                        "modelo": display_name,
                        "particion": split_name,
                        "n_registros": len(X_split),
                        "tasa_positiva_pct": round(float(y_split.mean() * 100), 2),
                        "accuracy_pct": round(metrics["accuracy"] * 100, 2),
                        "precision_pct": round(metrics["precision"] * 100, 2),
                        "recall_pct": round(metrics["recall"] * 100, 2),
                        "f1_score_pct": round(metrics["f1_score"] * 100, 2),
                        "roc_auc_pct": round(metrics["roc_auc"] * 100, 2),
                    }
                )

        return stability_rows

    def plot_confusion_matrices(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Path]:
        """Genera y guarda gráficos estilizados de matriz de confusión para cada modelo."""
        saved_plots = {}
        for key, result in evaluation_results.items():
            display_name = result["display_name"]
            cm = result["metrics"]["confusion_matrix"]
            tn = result["metrics"]["tn"]
            fp = result["metrics"]["fp"]
            fn = result["metrics"]["fn"]
            tp = result["metrics"]["tp"]

            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Pred: Sin Falla (0)", "Pred: Falla (1)"],
                yticklabels=["Real: Sin Falla (0)", "Real: Falla (1)"],
                ax=ax,
                annot_kws={"size": 14, "weight": "bold"},
            )

            ax.set_title(
                f"Matriz de Confusión - {display_name}\n(Test: Recall={result['metrics']['recall']*100:.2f}%, F1={result['metrics']['f1_score']*100:.2f}%)",
                fontsize=11,
                pad=12,
                fontweight="bold",
            )
            plt.tight_layout()

            plot_path = self.config.confusion_matrices_dir / f"matriz_confusion_{key}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            saved_plots[key] = plot_path

        return saved_plots

    def plot_comparative_roc_curves(
        self,
        y_test: pd.Series,
        evaluation_results: Dict[str, Dict[str, Any]],
    ) -> Path:
        """Genera la curva ROC comparativa de todos los modelos candidatos."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        colors = {
            "regresion_logistica": "#1f77b4",
            "random_forest": "#2ca02c",
            "xgboost": "#ff7f0e",
        }

        for key, result in evaluation_results.items():
            display_name = result["display_name"]
            y_proba = result["metrics"]["y_proba"]
            roc_auc = result["metrics"]["roc_auc"]

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            color = colors.get(key, "#333333")
            ax.plot(
                fpr,
                tpr,
                label=f"{display_name} (AUC = {roc_auc*100:.2f}%)",
                linewidth=2.2,
                color=color,
            )

        ax.plot([0, 1], [0, 1], "k--", label="Clasificador Aleatorio (AUC = 50.00%)", linewidth=1.2)
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Tasa de Verdaderos Positivos (Recall / Sensibilidad)", fontsize=11, fontweight="bold")
        ax.set_title("Curvas ROC Comparativas sobre Conjunto de Prueba (X_test)", fontsize=12, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=10, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        roc_plot_path = self.config.roc_curves_dir / "curvas_roc_comparativas.png"
        fig.savefig(roc_plot_path)
        plt.close(fig)

        return roc_plot_path
