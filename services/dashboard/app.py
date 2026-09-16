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
    load_prediction_history,
    check_api_health,
    get_api_models,
    set_active_model_via_api,
)
from services.dashboard.components import (
    apply_custom_styles,
    render_api_status_badge,
    render_kpi_cards,
    plot_probability_by_equipment,
    plot_risk_distribution_pie,
)
from services.dashboard.prediction import render_prediction_page
from services.dashboard.model_lab import render_model_lab_page
from services.dashboard.data_pipeline_page import render_data_pipeline_page


def main():
    st.set_page_config(
        page_title="GELATO PREDICT | Sistema de Mantenimiento 4.0",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    config = DashboardConfig(base_dir=BASE_DIR)

    # 1. Comprobar estado de FastAPI
    is_api_connected, health_data = check_api_health(config.api_base_url)
    api_models, active_model_backend = get_api_models(config.api_base_url) if is_api_connected else ([], "modelo_random_forest.joblib")

    # Cargar datos procesados base y resumen de evaluación
    df_full = load_dataset_modelado(config.processed_data_dir)
    eval_summary = load_evaluation_summary(config.results_dir)
    df_preds_final = eval_summary["predicciones_finales"]

    try:
        X_train, _, _, _, _, _ = load_split_datasets(config.processed_data_dir)
        feature_columns = list(X_train.columns)
    except Exception:
        feature_columns = []

    # 2. Barra Lateral (Sidebar Estilo AdminLTE / Next.js Pro)
    with st.sidebar:
        st.markdown(
            """
            <div class="admin-brand-box">
                <div class="admin-brand-icon">📊</div>
                <div class="admin-brand-text">
                    <div class="admin-brand-name">Ciencia de Datos</div>
                    <div class="admin-brand-version">v1.0.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Inicializar página activa en session_state
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "🏠 Panel General"

        nav_items = [
            ("🏠 Panel General", "panel"),
            ("📥 Carga y Pipeline", "pipeline"),
            ("🔮 Diagnóstico y Predicción", "prediccion"),
            ("📊 Análisis de Modelos", "analisis"),
            ("📋 Historial Operativo", "historial"),
            ("🧠 Laboratorio de Modelos", "laboratorio"),
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

        st.markdown('<div class="sidebar-section-title">⚙️ Motor de Inferencia (API)</div>', unsafe_allow_html=True)

        # Estado de conexión con FastAPI
        render_api_status_badge(
            is_connected=is_api_connected,
            active_model=active_model_backend,
            api_url=config.api_base_url,
        )

        # Selector de Modelo Activo
        if is_api_connected and api_models:
            model_options = {f"{m['name']} ({m['filename']})": m['filename'] for m in api_models}
            current_idx = 0
            for idx, (label, fname) in enumerate(model_options.items()):
                if fname == active_model_backend:
                    current_idx = idx
                    break

            selected_label = st.selectbox(
                "Modelo en Producción:",
                options=list(model_options.keys()),
                index=current_idx,
                help="El modelo seleccionado será ejecutado por el backend FastAPI en /predict.",
            )
            selected_fname = model_options[selected_label]

            if selected_fname != active_model_backend:
                if st.button("🔄 Aplicar en Producción", use_container_width=True):
                    if set_active_model_via_api(config.api_base_url, selected_fname):
                        st.success(f"Modelo cambiado a: {selected_fname}")
                        st.rerun()

        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #475569; text-align: center; margin-top: 25px; line-height: 1.4;">
                <strong>Área de Helados</strong><br>
                
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Renderizado según la opción seleccionada

    # ----------------------------------------------------
    # SECCIÓN 1: PANEL GENERAL (DASHBOARD)
    # ----------------------------------------------------
    if menu_option == "🏠 Panel General":
        st.markdown(f"<div class='main-header'>🏭 {config.app_title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-header'>{config.app_subtitle} — {config.app_description}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        total_equipos = df_full["Identificador_Equipo"].nunique() if not df_full.empty else (
            df_preds_final["Identificador_Equipo"].nunique() if not df_preds_final.empty else 0
        )

        if not df_preds_final.empty:
            prob_col = (
                "Probabilidad_Falla_7_Dias_Pct"
                if "Probabilidad_Falla_7_Dias_Pct" in df_preds_final.columns
                else "Prob_Falla_Random Forest (%)"
            )
            # Agrupar por equipo tomando su última observación evaluada
            df_latest_by_eq = (
                df_preds_final.sort_values(by="Fecha_Observacion")
                .groupby("Identificador_Equipo")
                .last()
                .reset_index()
            )
            criticos = int((df_latest_by_eq[prob_col] >= 60).sum())
            observacion = int(((df_latest_by_eq[prob_col] >= 30) & (df_latest_by_eq[prob_col] < 60)).sum())
            operativos = int((df_latest_by_eq[prob_col] < 30).sum())
        else:
            operativos, observacion, criticos = 0, 0, 0

        render_kpi_cards(
            total_equipos=total_equipos,
            equipos_bajo=operativos,
            equipos_medio=observacion,
            equipos_alto=criticos,
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

        # Tabla Completa de Predicciones con Filtros Interactivos
        st.markdown("---")
        st.subheader("📋 Explorador de Predicciones del Modelo Final (2,436 Observaciones)")
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
    # SECCIÓN 2: PREDICCIÓN (DIAGNÓSTICO EN TIEMPO REAL)
    # ----------------------------------------------------
    elif menu_option == "🔮 Diagnóstico y Predicción":
        render_prediction_page(
            config=config,
            feature_columns=feature_columns,
        )

    # ----------------------------------------------------
    # SECCIÓN 3: ANÁLISIS DE MODELOS
    # ----------------------------------------------------
    elif menu_option == "📊 Análisis de Modelos":
        st.markdown("<div class='main-header'>📊 Análisis Comparativo y Desempeño de Modelos</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Resultados formales de la evaluación de modelos sobre el conjunto de prueba independiente.</div>", unsafe_allow_html=True)

        st.subheader("🏆 Comparativa de Rendimiento en Conjunto de Prueba (X_test)")
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

        st.markdown("---")
        st.subheader("🛡️ Diagnóstico de Estabilidad y Control de Sobreajuste (Train / Valid / Test)")
        df_stab = eval_summary["diagnostico_estabilidad"]
        if not df_stab.empty:
            st.dataframe(df_stab, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📈 Comparativa Multi-Modelo por Fecha (Test Set)")
        df_all_models = pd.read_csv(config.evaluation_results_dir / "predicciones_todos_modelos_test.csv", encoding="utf-8-sig")
        if not df_all_models.empty:
            st.dataframe(
                df_all_models.head(20).rename(
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
    # SECCIÓN 4: HISTORIAL OPERATIVO
    # ----------------------------------------------------
    elif menu_option == "📋 Historial Operativo":
        st.markdown("<div class='main-header'>📋 Historial de Predicciones Operativas</div>", unsafe_allow_html=True)
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
                label="📥 Descargar Historial en CSV",
                data=csv_data,
                file_name="historial_predicciones_mantenimiento.csv",
                mime="text/csv",
            )
        else:
            st.info("ℹ️ Aún no hay predicciones en el historial. Realiza una predicción en la pestaña '🔮 Diagnóstico y Predicción' para registrarla.")

    # ----------------------------------------------------
    # SECCIÓN 5: LABORATORIO DE EXPERIMENTACIÓN / MODELOS
    # ----------------------------------------------------
    elif menu_option == "🧠 Laboratorio de Modelos":
        render_model_lab_page(config=config)

    # ----------------------------------------------------
    # SECCIÓN 6: CARGA DE DATOS Y PIPELINE
    # ----------------------------------------------------
    elif menu_option == "📥 Carga y Pipeline":
        render_data_pipeline_page(config=config)


if __name__ == "__main__":
    main()
