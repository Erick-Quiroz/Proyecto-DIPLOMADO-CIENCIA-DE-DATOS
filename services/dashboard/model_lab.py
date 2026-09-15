"""Laboratorio de Modelos: Experimentación, ajuste de hiperparámetros y promoción de modelos."""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import streamlit as st

from services.dashboard.config import DashboardConfig
from services.dashboard.utils import (
    load_split_datasets,
    save_experiment_to_csv,
    load_experiments_history,
    set_active_model_via_api,
)


def render_model_lab_page(config: DashboardConfig):
    """Renderiza el Laboratorio de Modelos para experimentación controlada y promoción de modelos."""
    st.markdown("<div class='main-header'>🧠 Laboratorio de Modelos de Machine Learning</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Ajusta hiperparámetros, reentrena sobre Train, valida en Validation y promueve modelos a producción en FastAPI.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="custom-info-box">
            🛡️ <strong>Garantía de Integridad Metodológica:</strong><br>
            El Laboratorio entrena <strong>estrictamente con particiones de Entrenamiento y Validación</strong>.<br>
            El conjunto de Prueba independiente permanece aislado para evitar sobreajuste y fuga de información.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cargar particiones
    try:
        X_train, X_valid, _, y_train, y_valid, _ = load_split_datasets(config.processed_data_dir)
    except Exception as e:
        st.error(f"Error al cargar particiones de datos: {e}")
        return

    # 1. Selector de Algoritmo Candidato
    col_sel1, col_sel2 = st.columns([1, 1.5])
    with col_sel1:
        model_choice = st.selectbox(
            "Selecciona el Algoritmo Candidato:",
            options=["Random Forest", "XGBoost", "Regresión Logística"],
            index=0,
        )
    with col_sel2:
        metric_priority = st.selectbox(
            "Métrica Principal de Decisión (Criterio de Negocio):",
            options=["Recall (Minimizar Fallas Omitidas)", "F1-Score (Balance)", "ROC-AUC (Discriminación)", "Accuracy"],
            index=0,
        )

    # 2. Configuración de Hiperparámetros según el modelo
    st.subheader(f"⚙️ Configuración de Hiperparámetros: {model_choice}")
    hyperparams = {}

    if model_choice == "Regresión Logística":
        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            hyperparams["C"] = st.slider("Parámetro de Regularización C", 0.01, 10.0, 1.0, 0.05)
            hyperparams["max_iter"] = st.slider("Máximo de Iteraciones", 100, 2000, 1000, 100)
        with col_hp2:
            hyperparams["solver"] = st.selectbox("Solver de Optimización", ["lbfgs", "liblinear", "saga", "newton-cg"], index=0)
            hyperparams["class_weight"] = st.selectbox("Balanceo de Clases", ["balanced", None], index=0)
        with col_hp3:
            hyperparams["penalty"] = st.selectbox("Penalización (penalty)", ["l2", "none"] if hyperparams["solver"] != "liblinear" else ["l2", "l1"], index=0)
            if hyperparams["penalty"] == "none":
                hyperparams["penalty"] = None
            hyperparams["random_state"] = 42

        st.caption("ℹ️ Nota: Regresión Logística incluye automáticamente `StandardScaler` encapsulado dentro del Pipeline.")

    elif model_choice == "Random Forest":
        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            hyperparams["n_estimators"] = st.slider("Número de Árboles (n_estimators)", 10, 300, 100, 10)
            hyperparams["max_depth"] = st.slider("Profundidad Máxima (max_depth)", 2, 30, 10, 1)
        with col_hp2:
            hyperparams["min_samples_split"] = st.slider("Muestras Mínimas para Split", 2, 20, 5, 1)
            hyperparams["min_samples_leaf"] = st.slider("Muestras Mínimas en Hoja", 1, 10, 2, 1)
        with col_hp3:
            hyperparams["max_features"] = st.selectbox("Variables Máximas (max_features)", ["sqrt", "log2", None], index=0)
            hyperparams["class_weight"] = st.selectbox("Ponderación de Clases", ["balanced", "balanced_subsample", None], index=0)
            hyperparams["random_state"] = 42
            hyperparams["n_jobs"] = -1

    elif model_choice == "XGBoost":
        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            hyperparams["n_estimators"] = st.slider("Número de Estimadores (n_estimators)", 10, 300, 100, 10)
            hyperparams["max_depth"] = st.slider("Profundidad Máxima (max_depth)", 2, 15, 5, 1)
            hyperparams["learning_rate"] = st.slider("Tasa de Aprendizaje (learning_rate)", 0.01, 0.5, 0.1, 0.01)
        with col_hp2:
            hyperparams["subsample"] = st.slider("Submuestra de Filas (subsample)", 0.5, 1.0, 0.8, 0.05)
            hyperparams["colsample_bytree"] = st.slider("Submuestra de Columnas (colsample_bytree)", 0.5, 1.0, 0.8, 0.05)
        with col_hp3:
            hyperparams["min_child_weight"] = st.slider("Peso Mínimo en Nodo (min_child_weight)", 1, 10, 1, 1)
            hyperparams["gamma"] = st.slider("Gamma (Reducción Mínima)", 0.0, 5.0, 0.0, 0.1)
            hyperparams["eval_metric"] = "logloss"
            hyperparams["random_state"] = 42
            hyperparams["n_jobs"] = -1

    # 3. Botón de Entrenamiento
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 ENTRENAR MODELO CON ESTOS HIPERPARÁMETROS", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo exclusivamente sobre X_train (12,412 registros)..."):
            start_time = time.time()

            if model_choice == "Regresión Logística":
                clean_params = {k: v for k, v in hyperparams.items() if v is not None}
                estimator = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(**clean_params)),
                ])
            elif model_choice == "Random Forest":
                estimator = RandomForestClassifier(**hyperparams)
            else:
                estimator = XGBClassifier(**hyperparams)

            estimator.fit(X_train, y_train)
            train_duration = time.time() - start_time

            # Inferencia sobre Validación
            y_proba_val = estimator.predict_proba(X_valid)[:, 1]

            st.session_state["lab_last_model"] = estimator
            st.session_state["lab_model_choice"] = model_choice
            st.session_state["lab_hyperparams"] = hyperparams
            st.session_state["lab_train_duration"] = train_duration
            st.session_state["lab_y_proba_val"] = y_proba_val
            st.session_state["lab_y_valid"] = y_valid
            st.session_state["lab_saved_path"] = None

            st.success(f"✅ ¡Entrenamiento completado en {train_duration:.3f} segundos!")

    # 4. Mostrar Resultados de Validación
    if "lab_last_model" in st.session_state:
        estimator = st.session_state["lab_last_model"]
        y_proba_val = st.session_state["lab_y_proba_val"]
        y_val = st.session_state["lab_y_valid"]
        duration = st.session_state["lab_train_duration"]
        current_model_name = st.session_state["lab_model_choice"]
        current_params = st.session_state["lab_hyperparams"]

        st.markdown("---")
        st.subheader("📊 Evaluación sobre Validación (X_valid - 2,668 registros)")

        # Simulador de Threshold
        col_th1, col_th2 = st.columns([1, 2])
        with col_th1:
            threshold = st.slider(
                "🎯 Umbral de Decisión (Threshold):",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Ajusta el corte de probabilidad para clasificar como alerta de falla (1).",
            )

        y_pred_val = (y_proba_val >= threshold).astype(int)

        acc = accuracy_score(y_val, y_pred_val)
        prec = precision_score(y_val, y_pred_val, zero_division=0)
        rec = recall_score(y_val, y_pred_val, zero_division=0)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        auc = roc_auc_score(y_val, y_proba_val)
        cm = confusion_matrix(y_val, y_pred_val)

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Recall (Sensibilidad)", f"{rec*100:.2f} %", help="Prioridad de mantenimiento")
        m_col2.metric("F1-Score", f"{f1*100:.2f} %")
        m_col3.metric("Precision", f"{prec*100:.2f} %")
        m_col4.metric("ROC-AUC", f"{auc*100:.2f} %")
        m_col5.metric("Accuracy", f"{acc*100:.2f} %")

        # Gráficos de Validación
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            cm_fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred: Normal (0)", "Pred: Falla (1)"],
                    y=["Real: Normal (0)", "Real: Falla (1)"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 18, "color": "black"},
                )
            )
            cm_fig.update_layout(
                title=f"Matriz de Confusión en Validación (Threshold = {threshold:.2f})",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        with col_g2:
            th_range = np.linspace(0.1, 0.9, 17)
            precs, recs, f1s = [], [], []
            for th in th_range:
                yp = (y_proba_val >= th).astype(int)
                precs.append(precision_score(y_val, yp, zero_division=0) * 100)
                recs.append(recall_score(y_val, yp, zero_division=0) * 100)
                f1s.append(f1_score(y_val, yp, zero_division=0) * 100)

            tradeoff_fig = go.Figure()
            tradeoff_fig.add_trace(go.Scatter(x=th_range, y=recs, mode="lines+markers", name="Recall (%)", line=dict(color="#28a745", width=2.5)))
            tradeoff_fig.add_trace(go.Scatter(x=th_range, y=f1s, mode="lines+markers", name="F1-Score (%)", line=dict(color="#3b82f6", width=2.5)))
            tradeoff_fig.add_trace(go.Scatter(x=th_range, y=precs, mode="lines+markers", name="Precision (%)", line=dict(color="#ffc107", width=2.5)))
            tradeoff_fig.add_vline(x=threshold, line_dash="dash", line_color="#dc3545", annotation_text=f"Actual: {threshold:.2f}")

            tradeoff_fig.update_layout(
                title="Trade-off en Validación (Precision vs Recall vs F1)",
                xaxis_title="Threshold",
                yaxis_title="Rendimiento (%)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            )
            st.plotly_chart(tradeoff_fig, use_container_width=True)

        # 5. Guardar Modelo y Promover a Producción
        st.markdown("### 💾 Persistencia y Promoción de Modelos")
        col_sv1, col_sv2 = st.columns([2, 1])

        with col_sv1:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_tag = current_model_name.lower().replace(" ", "_")
            default_version_name = f"laboratorio_{clean_tag}_{timestamp_str}"
            save_name_input = st.text_input(
                "Nombre de versión para guardar en models/laboratorio/:",
                value=default_version_name,
            )

        with col_sv2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 GUARDAR MODELO", use_container_width=True):
                clean_filename = f"{save_name_input.strip()}.joblib"
                save_path = config.lab_models_dir / clean_filename

                joblib.dump(estimator, save_path)
                st.session_state["lab_saved_path"] = clean_filename

                # Registrar experimento
                exp_record = {
                    "id_experimento": f"EXP_{timestamp_str}",
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "modelo": current_model_name,
                    "archivo_guardado": clean_filename,
                    "accuracy_val_pct": round(acc * 100, 2),
                    "precision_val_pct": round(prec * 100, 2),
                    "recall_val_pct": round(rec * 100, 2),
                    "f1_val_pct": round(f1 * 100, 2),
                    "roc_auc_val_pct": round(auc * 100, 2),
                    "threshold_utilizado": round(threshold, 2),
                    "tiempo_entrenamiento_seg": round(duration, 4),
                    "hiperparametros": str(current_params),
                }
                save_experiment_to_csv(config, exp_record)
                st.success(f"🎉 Modelo guardado como `{clean_filename}` en `models/laboratorio/`.")

        # Botón para Establecer como Modelo Activo en FastAPI
        if st.session_state.get("lab_saved_path"):
            saved_fname = st.session_state["lab_saved_path"]
            st.markdown(
                f"""
                <div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 6px; margin: 15px 0;">
                    <strong>⭐ Promoción a Modelo Activo en Producción:</strong><br>
                    ¿Deseas activar <code>{saved_fname}</code> en el backend de FastAPI para las predicciones en vivo?
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("⭐ ESTABLECER COMO MODELO ACTIVO EN FASTAPI", type="secondary"):
                success = set_active_model_via_api(config.api_base_url, saved_fname)
                if success:
                    st.success(f"🚀 ¡El modelo `{saved_fname}` ahora es el MODELO ACTIVO en FastAPI!")
                    st.balloons()
                else:
                    # Si la API no está corriendo, persistir directamente en archivo de config
                    config.set_active_model_filename(saved_fname)
                    st.info(f"Modelo configurado como activo en disco: `{saved_fname}`.")

    # 6. Historial de Experimentos
    st.markdown("---")
    st.subheader("📋 Registro de Experimentos de Laboratorio")
    df_exp = load_experiments_history(config)

    if not df_exp.empty:
        st.dataframe(
            df_exp.sort_values(by="fecha", ascending=False).rename(
                columns={
                    "fecha": "Fecha/Hora",
                    "modelo": "Algoritmo",
                    "archivo_guardado": "Archivo Guardado",
                    "recall_val_pct": "Recall (%)",
                    "f1_val_pct": "F1 (%)",
                    "precision_val_pct": "Precision (%)",
                    "roc_auc_val_pct": "ROC-AUC (%)",
                    "threshold_utilizado": "Threshold",
                    "tiempo_entrenamiento_seg": "Tiempo (s)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Aún no hay experimentos guardados. Entrena y guarda un modelo para registrar la trazabilidad.")
