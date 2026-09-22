"""Punto de entrada principal para el Dashboard de Mantenimiento Predictivo con Streamlit."""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Configurar ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.dashboard.config import DashboardConfig
from services.dashboard.utils import (
    load_dataset_modelado,
    load_split_datasets,
    load_evaluation_summary,
    load_temporal_partition_summary,
    load_prediction_history,
    check_api_health,
)
from services.dashboard.components import (
    apply_custom_styles,
    render_kpi_cards,
    plot_probability_by_equipment,
    plot_risk_distribution_pie,
    render_temporal_partition_section,
)
from services.dashboard.prediction import render_prediction_page
from services.dashboard.model_lab import render_model_lab_page
from services.dashboard.data_pipeline_page import render_data_pipeline_page
from services.dashboard.auth import check_dashboard_auth, render_sidebar_user_profile


def main():
    st.set_page_config(
        page_title="GELATO PREDICT | Sistema de Mantenimiento 4.0",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    config = DashboardConfig(base_dir=BASE_DIR)

    # 1. Control de Autenticación Obligatorio (Gatekeeper de Seguridad)
    if not check_dashboard_auth(config):
        st.stop()

    # 2. Comprobar estado de FastAPI
    is_api_connected, _ = check_api_health(config.api_base_url)

    # Cargar datos procesados base y resumen de evaluación
    df_full = load_dataset_modelado(config.processed_data_dir)
    eval_summary = load_evaluation_summary(config.results_dir)
    df_preds_final = eval_summary["predicciones_finales"]
    df_partition = load_temporal_partition_summary(config.base_dir, df_full=df_full)

    try:
        X_train, _, _, _, _, _ = load_split_datasets(config.processed_data_dir)
        feature_columns = list(X_train.columns)
    except Exception:
        feature_columns = []

    # 3. Barra Lateral (Sidebar Estilo AdminLTE / Next.js Pro sin íconos)
    with st.sidebar:
        st.markdown(
            """
            <div class="admin-brand-box" style="padding: 12px 14px;">
                <div class="admin-brand-text">
                    <div class="admin-brand-name">Ciencia de Datos</div>
                    <div class="admin-brand-version">v1.0.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mostrar perfil de usuario autenticado y botón de cerrar sesión
        render_sidebar_user_profile(config)

        # Inicializar página activa en session_state
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Panel General"

        nav_items = [
            ("Panel General", "panel"),
            ("Carga y Pipeline", "pipeline"),
            ("Diagnóstico y Predicción", "prediccion"),
            ("Análisis de Modelos", "analisis"),
            ("Historial Operativo", "historial"),
            ("Laboratorio de Modelos", "laboratorio"),
        ]

        for label, key_id in nav_items:
            is_active = (st.session_state["current_page"] == label)
            if st.button(
                label,
                key=f"sidebar_btn_{key_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["current_page"] = label
                st.rerun()

        menu_option = st.session_state["current_page"]

        # Footer con estado de conexión limpio al final del Sidebar
        if is_api_connected:
            st.markdown(
                """
                <div style="font-size: 0.78rem; font-weight: 700; color: #10B981; text-align: center; margin-top: 60px; padding: 8px 12px; background: rgba(16, 185, 129, 0.08); border-radius: 6px; letter-spacing: 0.03em;">
                    ● BACKEND ONLINE
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-size: 0.78rem; font-weight: 700; color: #EF4444; text-align: center; margin-top: 60px; padding: 8px 12px; background: rgba(239, 68, 68, 0.08); border-radius: 6px; letter-spacing: 0.03em;">
                    ● BACKEND OFFLINE
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #475569; text-align: center; margin-top: 15px; line-height: 1.4;">
                <strong>Área de Helados</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Renderizado según la opción seleccionada

    # ----------------------------------------------------
    # SECCIÓN 1: PANEL GENERAL (DASHBOARD)
    # ----------------------------------------------------
    if menu_option == "Panel General":
        st.markdown(f"<div class='main-header'>{config.app_title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-header'>{config.app_subtitle} — {config.app_description}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        total_equipos = df_full["Identificador_Equipo"].nunique() if not df_full.empty else (
            df_preds_final["Identificador_Equipo"].nunique() if not df_preds_final.empty else 0
        )

        if not df_preds_final.empty:
            if "Probabilidad_Falla_7_Dias" in df_preds_final.columns:
                prob_series = df_preds_final["Probabilidad_Falla_7_Dias"]
            elif "Probabilidad_Falla_7_Dias_Pct" in df_preds_final.columns:
                prob_series = df_preds_final["Probabilidad_Falla_7_Dias_Pct"] / 100.0
            elif "Prob_Falla_Random Forest (%)" in df_preds_final.columns:
                prob_series = df_preds_final["Prob_Falla_Random Forest (%)"] / 100.0
            else:
                prob_series = df_preds_final.iloc[:, 0]

            total_obs = len(df_preds_final)
            criticos = int((prob_series >= 0.60).sum())
            observacion = int(((prob_series >= 0.30) & (prob_series < 0.60)).sum())
            operativos = int((prob_series < 0.30).sum())
        else:
            total_obs, operativos, observacion, criticos = 0, 0, 0, 0

        render_kpi_cards(
            total_equipos=total_equipos,
            equipos_bajo=operativos,
            equipos_medio=observacion,
            equipos_alto=criticos,
            total_observaciones=total_obs,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Gráficos de Resumen
        col_g1, col_g2 = st.columns([1.4, 1])
        with col_g1:
            if not df_preds_final.empty:
                fig_bar = plot_probability_by_equipment(df_preds_final)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No hay datos de predicción disponibles para graficar.")

        with col_g2:
            if not df_preds_final.empty:
                fig_pie = plot_risk_distribution_pie(df_preds_final, config)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No hay datos de distribución de riesgo disponibles.")

        # Sección del Esquema de Partición Temporal Cronológica (Dinámica con purga 7D)
        render_temporal_partition_section(df_partition)

        # Tabla Completa de Predicciones con Filtros Interactivos
        st.markdown("---")
        st.subheader("Explorador de Predicciones del Modelo Final")
        st.caption("Visualiza y filtra todas las evaluaciones del conjunto de prueba independiente (`predicciones_modelo_final.csv`).")

        if not df_preds_final.empty:
            col_flt1, col_flt2, col_flt3 = st.columns(3)
            with col_flt1:
                eq_filter = st.multiselect("Filtrar por Equipo:", options=sorted(df_preds_final["Nombre_Equipo"].unique()))
            with col_flt2:
                estado_filter = st.selectbox("Filtrar por Tipo de Registro:", ["Todos los Registros", "Solo Días con Falla Real Confirmada (1)", "Solo Días con Alerta Emitida (1)"])
            with col_flt3:
                rango_prob = st.slider("Filtrar por % de Probabilidad:", 0.0, 100.0, (0.0, 100.0), step=5.0)

            df_view = df_preds_final.copy()
            if eq_filter:
                df_view = df_view[df_view["Nombre_Equipo"].isin(eq_filter)]
            if estado_filter == "Solo Días con Falla Real Confirmada (1)":
                df_view = df_view[df_view["Falla_Real"] == 1]
            elif estado_filter == "Solo Días con Alerta Emitida (1)":
                df_view = df_view[df_view["Prediccion"] == 1]

            df_view = df_view[
                (df_view["Probabilidad_Falla_7_Dias_Pct"] >= rango_prob[0])
                & (df_view["Probabilidad_Falla_7_Dias_Pct"] <= rango_prob[1])
            ]

            st.markdown(f"**Mostrando {len(df_view):,d} observaciones filtradas:**")

            cols_show = [
                col for col in [
                    "Identificador_Equipo",
                    "Nombre_Equipo",
                    "Tipo_Equipo",
                    "Línea",
                    "Fecha_Observacion",
                    "Falla_Real",
                    "Prediccion",
                    "Probabilidad_Falla_7_Dias_Pct",
                    "Criticidad",
                ] if col in df_view.columns
            ]
            st.dataframe(
                df_view[cols_show].sort_values(by="Probabilidad_Falla_7_Dias_Pct", ascending=False).rename(
                    columns={
                        "Identificador_Equipo": "ID Equipo",
                        "Nombre_Equipo": "Equipo",
                        "Tipo_Equipo": "Tipo",
                        "Línea": "Línea de Producción",
                        "Fecha_Observacion": "Fecha Telemetría",
                        "Falla_Real": "Falla Real (7D)",
                        "Prediccion": "Predicción",
                        "Probabilidad_Falla_7_Dias_Pct": "Probabilidad de Falla (%)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ----------------------------------------------------
    # SECCIÓN 2: CARGA Y PIPELINE
    # ----------------------------------------------------
    elif menu_option == "Carga y Pipeline":
        render_data_pipeline_page(config=config)

    # ----------------------------------------------------
    # SECCIÓN 3: PREDICCIÓN (DIAGNÓSTICO EN TIEMPO REAL)
    # ----------------------------------------------------
    elif menu_option == "Diagnóstico y Predicción":
        render_prediction_page(
            config=config,
            feature_columns=feature_columns,
        )

    # ----------------------------------------------------
    # SECCIÓN 4: ANÁLISIS DE MODELOS
    # ----------------------------------------------------
    elif menu_option == "Análisis de Modelos":
        st.markdown("<div class='main-header'>Análisis Comparativo y Desempeño de Modelos</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Consulta las métricas formales en Validación (X_valid), Prueba (X_test) y Diagnóstico de Estabilidad.</div>", unsafe_allow_html=True)

        tab_valid, tab_test, tab_stab, tab_plots, tab_detail = st.tabs([
            "🧪 Validación (X_valid)",
            "🎯 Prueba (X_test)",
            "⚖️ Diagnóstico de Estabilidad (Train/Val/Test)",
            "📊 Curvas ROC y Matrices",
            "📋 Predicciones por Equipo (Test)",
        ])

        # TAB 1: VALIDACIÓN (X_valid)
        with tab_valid:
            st.markdown("### 🧪 Evaluación en Conjunto de Validación (2,465 Registros | 289 Fallas)")
            st.caption("Fase 7.4 CRISP-DM: Partición temporal utilizada para selección metodológica y calibración.")
            
            df_val = eval_summary.get("tabla_validacion", pd.DataFrame())
            if not df_val.empty:
                val_cols = ["modelo", "accuracy_pct", "precision_pct", "recall_pct", "f1_score_pct", "roc_auc_pct"]
                available_cols = [c for c in val_cols if c in df_val.columns]
                val_display = df_val[available_cols].rename(
                    columns={
                        "modelo": "Modelo",
                        "accuracy_pct": "Accuracy (%)",
                        "precision_pct": "Precision (%)",
                        "recall_pct": "Recall (%)",
                        "f1_score_pct": "F1-Score (%)",
                        "roc_auc_pct": "ROC-AUC (%)",
                    }
                )
                st.dataframe(
                    val_display.style.format({
                        "Accuracy (%)": "{:.2f} %",
                        "Precision (%)": "{:.2f} %",
                        "Recall (%)": "{:.2f} %",
                        "F1-Score (%)": "{:.2f} %",
                        "ROC-AUC (%)": "{:.2f} %",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No se encontró la tabla de validación.")

            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Recall en Validación", "97.58 %", "Random Forest / Reg. Logística")
            with kpi2:
                st.metric("F1-Score en Validación", "84.27 %", "XGBoost")
            with kpi3:
                st.metric("ROC-AUC en Validación", "99.25 %", "XGBoost")

        # TAB 2: PRUEBA (X_test)
        with tab_test:
            st.markdown("### 🎯 Evaluación en Conjunto de Prueba Independiente (2,436 Registros | 310 Fallas)")
            st.caption("Fase 7.5 CRISP-DM: Evaluación final sobre el horizonte futuro independiente.")

            df_comp = eval_summary["tabla_comparacion"]
            if not df_comp.empty:
                st.dataframe(
                    df_comp.style.format({
                        "Accuracy": "{:.2%}",
                        "Precision": "{:.2%}",
                        "Recall": "{:.2%}",
                        "F1-Score": "{:.2%}",
                        "ROC-AUC": "{:.2%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Recall en Prueba", "99.35 %", "Random Forest (2 FN)")
            with kpi2:
                st.metric("F1-Score en Prueba", "86.86 %", "XGBoost")
            with kpi3:
                st.metric("ROC-AUC en Prueba", "99.29 %", "Regresión Logística")

        # TAB 3: ESTABILIDAD
        with tab_stab:
            st.markdown("### ⚖️ Diagnóstico de Estabilidad y Control de Sobreajuste (Train / Valid / Test)")
            st.caption("Comparativa cruzada entre particiones cronológicas para validar generalización.")
            df_stab = eval_summary["diagnostico_estabilidad"]
            if not df_stab.empty:
                st.dataframe(
                    df_stab.rename(
                        columns={
                            "id_modelo": "ID",
                            "modelo": "Modelo",
                            "particion": "Partición",
                            "n_registros": "N° Filas",
                            "tasa_positiva_pct": "Tasa Fallas (%)",
                            "accuracy_pct": "Accuracy (%)",
                            "precision_pct": "Precision (%)",
                            "recall_pct": "Recall (%)",
                            "f1_score_pct": "F1-Score (%)",
                            "roc_auc_pct": "ROC-AUC (%)",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # TAB 4: GRÁFICOS
        with tab_plots:
            st.markdown("### 📊 Curvas ROC y Comparativa Visual de Modelos")
            col_roc, col_bar = st.columns(2)
            roc_path = config.evaluation_results_dir / "curvas_roc" / "curvas_roc_comparativas.png"
            table_img_path = config.evaluation_results_dir / "tabla_comparacion_modelos.png"

            with col_roc:
                if roc_path.exists():
                    st.image(str(roc_path), caption="Curvas ROC Comparativas (Test)", use_container_width=True)
            with col_bar:
                if table_img_path.exists():
                    st.image(str(table_img_path), caption="Comparativa Gráfica de Rendimiento", use_container_width=True)

            st.markdown("#### Matrices de Confusión Individuales (Conjunto de Prueba)")
            cm_cols = st.columns(3)
            cm_models = [
                ("random_forest", "Random Forest", cm_cols[0]),
                ("xgboost", "XGBoost", cm_cols[1]),
                ("regresion_logistica", "Regresión Logística", cm_cols[2]),
            ]
            for key, name, col in cm_models:
                cm_path = config.evaluation_results_dir / "matrices_confusion" / f"matriz_confusion_{key}.png"
                with col:
                    if cm_path.exists():
                        st.image(str(cm_path), caption=f"Matriz: {name}", use_container_width=True)

        # TAB 5: PREDICCIONES MULTI-MODELO
        with tab_detail:
            st.markdown("### 📋 Predicciones Multi-Modelo por Equipo (Test Set)")
            df_all_models = pd.read_csv(config.evaluation_results_dir / "predicciones_todos_modelos_test.csv", encoding="utf-8-sig")
            if not df_all_models.empty:
                st.dataframe(
                    df_all_models.rename(
                        columns={
                            "Probabilidad_Falla_Regresión Logística (%)": "% Reg. Logística",
                            "Probabilidad_Falla_Random Forest (%)": "% Random Forest",
                            "Probabilidad_Falla_XGBoost (%)": "% XGBoost",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # ----------------------------------------------------
    # SECCIÓN 5: HISTORIAL OPERATIVO
    # ----------------------------------------------------
    elif menu_option == "Historial Operativo":
        st.markdown("<div class='main-header'>Historial de Predicciones Operativas</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Consulta y filtra todas las evaluaciones predictivas registradas en la planta.</div>", unsafe_allow_html=True)

        df_hist = load_prediction_history(config)

        if not df_hist.empty:
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                equipos_filter = st.multiselect("Filtrar por Equipo:", options=sorted(df_hist["Equipo"].unique()))
            with col_f2:
                modelos_filter = st.multiselect("Filtrar por Modelo:", options=sorted(df_hist["Modelo"].unique()))
            with col_f3:
                riesgos_filter = st.multiselect("Filtrar por Nivel de Riesgo:", options=sorted(df_hist["Riesgo"].unique()))

            df_filtered = df_hist.copy()
            if equipos_filter:
                df_filtered = df_filtered[df_filtered["Equipo"].isin(equipos_filter)]
            if modelos_filter:
                df_filtered = df_filtered[df_filtered["Modelo"].isin(modelos_filter)]
            if riesgos_filter:
                df_filtered = df_filtered[df_filtered["Riesgo"].isin(riesgos_filter)]

            st.dataframe(
                df_filtered.sort_values(by="Fecha", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            csv_data = df_filtered.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="Descargar Historial en CSV",
                data=csv_data,
                file_name="historial_predicciones_mantenimiento.csv",
                mime="text/csv",
            )
        else:
            st.info("Aún no hay predicciones en el historial. Realiza una predicción en la pestaña 'Diagnóstico y Predicción' para registrarla.")

    # ----------------------------------------------------
    # SECCIÓN 6: LABORATORIO DE EXPERIMENTACIÓN / MODELOS
    # ----------------------------------------------------
    elif menu_option == "Laboratorio de Modelos":
        render_model_lab_page(config=config)


if __name__ == "__main__":
    main()
