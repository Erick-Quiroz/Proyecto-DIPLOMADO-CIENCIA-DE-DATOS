"""Utilidades de carga de datos, cliente HTTP para FastAPI y persistencia."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests
import joblib
import pandas as pd
import streamlit as st

from services.dashboard.config import DashboardConfig


# =====================================================================
# CLIENTE HTTP PARA COMUNICACIÓN CON FASTAPI
# =====================================================================

def check_api_health(api_url: str) -> Tuple[bool, Dict[str, Any]]:
    """Comprueba el estado de salud de FastAPI mediante GET /health."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=2.0)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"Status code {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def get_api_models(api_url: str) -> Tuple[List[Dict[str, Any]], str]:
    """Obtiene la lista de modelos y el modelo activo desde GET /models."""
    try:
        resp = requests.get(f"{api_url}/models", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models", []), data.get("active_model", "")
        return [], ""
    except Exception:
        return [], ""


def predict_via_api(
    api_url: str,
    equipo: str,
    features: Dict[str, float],
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Envía la solicitud de inferencia al endpoint POST /predict de FastAPI."""
    payload = {
        "equipo": equipo,
        "features": features,
        "model_override": model_override,
    }
    resp = requests.post(f"{api_url}/predict", json=payload, timeout=5.0)
    if resp.status_code == 200:
        return resp.json()
    else:
        try:
            err_data = resp.json()
            detail = err_data.get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Error en FastAPI ({resp.status_code}): {detail}")


def set_active_model_via_api(api_url: str, model_filename: str) -> bool:
    """Promueve un modelo como activo en producción mediante POST /models/set-active."""
    payload = {"model_filename": model_filename}
    try:
        resp = requests.post(f"{api_url}/models/set-active", json=payload, timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# =====================================================================
# CARGA DE DATOS LOCALES CON CACHÉ
# =====================================================================

@st.cache_data(show_spinner=False)
def load_dataset_modelado(processed_data_dir: Path) -> pd.DataFrame:
    """Carga con caché el dataset consolidado de modelado con metadatos de planta."""
    path = processed_data_dir / "dataset_modelado.csv"
    if not path.exists():
        st.error(f"Archivo no encontrado: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_split_datasets(
    processed_data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Carga los conjuntos de partición cronológica (Train, Valid, Test)."""
    x_tr = pd.read_csv(processed_data_dir / "X_train.csv", encoding="utf-8-sig")
    y_tr = pd.read_csv(processed_data_dir / "y_train.csv", encoding="utf-8-sig").squeeze().astype(int)
    x_val = pd.read_csv(processed_data_dir / "X_valid.csv", encoding="utf-8-sig")
    y_val = pd.read_csv(processed_data_dir / "y_valid.csv", encoding="utf-8-sig").squeeze().astype(int)
    x_te = pd.read_csv(processed_data_dir / "X_test.csv", encoding="utf-8-sig")
    y_te = pd.read_csv(processed_data_dir / "y_test.csv", encoding="utf-8-sig").squeeze().astype(int)
    return x_tr, x_val, x_te, y_tr, y_val, y_te


@st.cache_data(show_spinner=False)
def load_evaluation_summary(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Carga los resultados generados durante la evaluación de modelos."""
    eval_dir = results_dir / "evaluation"
    comp_path = eval_dir / "tabla_comparacion_modelos.csv"
    metrics_path = eval_dir / "metricas_modelos.csv"
    preds_final_path = eval_dir / "predicciones_modelo_final.csv"
    stability_path = eval_dir / "diagnostico_estabilidad.csv"

    df_comp = pd.read_csv(comp_path, encoding="utf-8-sig") if comp_path.exists() else pd.DataFrame()
    df_metrics = pd.read_csv(metrics_path, encoding="utf-8-sig") if metrics_path.exists() else pd.DataFrame()
    df_preds_final = pd.read_csv(preds_final_path, encoding="utf-8-sig") if preds_final_path.exists() else pd.DataFrame()
    df_stability = pd.read_csv(stability_path, encoding="utf-8-sig") if stability_path.exists() else pd.DataFrame()

    df_val = pd.DataFrame()
    if not df_stability.empty and "particion" in df_stability.columns:
        val_mask = df_stability["particion"].isin(["Valid", "Validacion", "Validación"])
        df_val = df_stability[val_mask].copy().reset_index(drop=True)

    return {
        "tabla_comparacion": df_comp,
        "tabla_validacion": df_val,
        "metricas_modelos": df_metrics,
        "predicciones_finales": df_preds_final,
        "diagnostico_estabilidad": df_stability,
    }


@st.cache_data(show_spinner=False)
def load_temporal_partition_summary(base_dir: Path, df_full: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Carga o calcula dinámicamente el resumen de particiones temporales (Entrenamiento, Validación, Prueba)."""
    # 1. Intentar cargar desde tablas_preparacion o reports
    table_paths = [
        base_dir / "tablas_preparacion" / "resumen_particion_temporal.csv",
        base_dir / "reports" / "tables" / "preparacion" / "resumen_particion_temporal.csv",
    ]
    for p in table_paths:
        if p.exists():
            try:
                df_res = pd.read_csv(p, encoding="utf-8-sig")
                if not df_res.empty and "Conjunto" in df_res.columns:
                    return df_res
            except Exception:
                pass

    # 2. Si no existe o se provee df_full, calcularlo dinámicamente
    if df_full is not None and not df_full.empty:
        date_col = "Fecha_Observacion" if "Fecha_Observacion" in df_full.columns else "Fecha"
        split_col = "Conjunto_Temporal" if "Conjunto_Temporal" in df_full.columns else "Particion"
        target_col = "Falla_En_Los_Siguientes_Siete_Dias" if "Falla_En_Los_Siguientes_Siete_Dias" in df_full.columns else "Falla_Real"

        if split_col in df_full.columns and date_col in df_full.columns:
            df_work = df_full.copy()
            df_work[date_col] = pd.to_datetime(df_work[date_col])
            records = []
            order = ["Entrenamiento", "Validación", "Prueba"]
            for conj in order:
                prefix = conj[:4]
                mask = df_work[split_col].astype(str).str.startswith(prefix)
                df_sub = df_work[mask]
                if not df_sub.empty:
                    f0 = int((df_sub[target_col] == 0).sum()) if target_col in df_sub.columns else 0
                    f1 = int((df_sub[target_col] == 1).sum()) if target_col in df_sub.columns else 0
                    tot = len(df_sub)
                    tasa = round((f1 / tot) * 100, 2) if tot > 0 else 0.0
                    records.append({
                        "Conjunto": conj,
                        "Fecha_Inicio": str(df_sub[date_col].min().date()),
                        "Fecha_Fin": str(df_sub[date_col].max().date()),
                        "Total_Registros": tot,
                        "Fallas_0": f0,
                        "Fallas_1": f1,
                        "Tasa_Falla_Pct": tasa,
                    })
            if records:
                return pd.DataFrame(records)

    return pd.DataFrame()



# =====================================================================
# HISTORIAL Y EXPERIMENTOS
# =====================================================================

def save_prediction_to_history(config: DashboardConfig, record: Dict[str, Any]) -> Path:
    """Registra de forma persistente una predicción en results/predictions/."""
    history_path = config.predictions_dir / config.history_file
    df_new = pd.DataFrame([record])

    if history_path.exists():
        df_existing = pd.read_csv(history_path, encoding="utf-8-sig")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(history_path, index=False, encoding="utf-8-sig")
    return history_path


def load_prediction_history(config: DashboardConfig) -> pd.DataFrame:
    """Carga el historial persistente de predicciones."""
    history_path = config.predictions_dir / config.history_file
    if history_path.exists():
        return pd.read_csv(history_path, encoding="utf-8-sig")
    return pd.DataFrame()


def save_experiment_to_csv(config: DashboardConfig, exp_record: Dict[str, Any]) -> Path:
    """Registra de forma persistente un experimento del Laboratorio de Modelos."""
    exp_path = config.modeling_results_dir / config.experiments_file
    df_new = pd.DataFrame([exp_record])

    if exp_path.exists():
        df_existing = pd.read_csv(exp_path, encoding="utf-8-sig")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(exp_path, index=False, encoding="utf-8-sig")
    return exp_path


def load_experiments_history(config: DashboardConfig) -> pd.DataFrame:
    """Carga el registro histórico de experimentos del laboratorio."""
    exp_path = config.modeling_results_dir / config.experiments_file
    if exp_path.exists():
        return pd.read_csv(exp_path, encoding="utf-8-sig")
    return pd.DataFrame()
