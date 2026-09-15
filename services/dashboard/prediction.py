"""Módulo de inferencia operativa para predicción de fallas comunicándose con FastAPI."""

from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import streamlit as st

from services.dashboard.config import DashboardConfig
from services.dashboard.utils import (
    load_dataset_modelado,
    load_evaluation_summary,
    predict_via_api,
    save_prediction_to_history,
)
from services.dashboard.components import render_prediction_result_box, plot_equipment_timeline


def render_prediction_page(
    config: DashboardConfig,
    feature_columns: List[str],
):
    """Renderiza la interfaz operativa de predicción de fallas con histórico temporal y FastAPI."""
    st.markdown("<div class='main-header'>🔮 Diagnóstico y Predicción de Fallas</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Selecciona un equipo industrial y una fecha para evaluar su riesgo predictivo o simular escenarios en FastAPI.</div>",
        unsafe_allow_html=True,
    )

    df_full = load_dataset_modelado(config.processed_data_dir)
    eval_summary = load_evaluation_summary(config.results_dir)
    df_preds_final = eval_summary["predicciones_finales"]

    if df_full.empty:
        st.error("No se pudo cargar el dataset base para inferencia.")
        return

    # 1. Selector de Equipo
    equipos_unicos = (
        df_full[["Identificador_Equipo", "Nombre_Equipo", "Tipo_Equipo", "Línea", "Proceso"]]
        .drop_duplicates(subset=["Identificador_Equipo"])
        .sort_values(by="Nombre_Equipo")
    )

    opciones_equipos = [
        f"EQ-{row['Identificador_Equipo']} - {row['Nombre_Equipo']} ({row['Tipo_Equipo']})"
        for _, row in equipos_unicos.iterrows()
    ]
    mapa_ids = {
        f"EQ-{row['Identificador_Equipo']} - {row['Nombre_Equipo']} ({row['Tipo_Equipo']})": row["Identificador_Equipo"]
        for _, row in equipos_unicos.iterrows()
    }

    col_eq1, col_eq2 = st.columns([1.5, 1])
    with col_eq1:
        selected_option = st.selectbox(
            "🏭 Selecciona el Equipo a Diagnosticar:",
            options=opciones_equipos,
            index=0,
        )
        selected_id = mapa_ids[selected_option]

    # Filtrar datos de este equipo
    df_equipo_full = df_full[df_full["Identificador_Equipo"] == selected_id].sort_values(
        by="Fecha_Observacion", ascending=False
    )
    df_equipo_preds = (
        df_preds_final[df_preds_final["Identificador_Equipo"] == selected_id].sort_values(
            by="Fecha_Observacion", ascending=False
        )
        if not df_preds_final.empty
        else pd.DataFrame()
    )

    # 2. Selector de Fecha de Observación
    fechas_disponibles = list(df_equipo_full["Fecha_Observacion"].unique())

    # Buscar fecha con máxima probabilidad en test para sugerir
    max_risk_date = fechas_disponibles[0]
    if not df_equipo_preds.empty:
        idx_max = df_equipo_preds["Probabilidad_Falla_7_Dias_Pct"].idxmax()
        max_risk_date = df_equipo_preds.loc[idx_max, "Fecha_Observacion"]

    with col_eq2:
        selected_date = st.selectbox(
            "📅 Selecciona la Fecha de Telemetría:",
            options=fechas_disponibles,
            index=0,
            help="Selecciona cualquier fecha del historial para cargar los sensores exactos registrados en ese día.",
        )

    # Botones de acceso rápido a fechas críticas
    st.markdown("**Atajos de Fecha para este Equipo:**")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button(f"🚨 Día de Mayor Riesgo ({max_risk_date})", use_container_width=True):
            selected_date = max_risk_date
            st.rerun()
    with btn_col2:
        if st.button(f"🟢 Día Normal Inicial ({fechas_disponibles[-1]})", use_container_width=True):
            selected_date = fechas_disponibles[-1]
            st.rerun()
    with btn_col3:
        if st.button(f"📅 Última Fecha ({fechas_disponibles[0]})", use_container_width=True):
            selected_date = fechas_disponibles[0]
            st.rerun()

    # Obtener el registro correspondiente a la fecha seleccionada
    df_row_selected = df_equipo_full[df_equipo_full["Fecha_Observacion"] == selected_date]
    active_row = df_row_selected.iloc[0] if not df_row_selected.empty else df_equipo_full.iloc[0]

    # Metadatos del equipo
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Tipo de Equipo", str(active_row.get("Tipo_Equipo", "N/A")))
    with col_info2:
        st.metric("Línea Productiva", str(active_row.get("Línea", "N/A")))
    with col_info3:
        st.metric("Proceso", str(active_row.get("Proceso", "N/A")))
    with col_info4:
        st.metric("Fecha Telemetría", str(selected_date))

    # 3. Línea de tiempo interactiva de la probabilidad de falla
    if not df_equipo_preds.empty:
        st.markdown("---")
        st.subheader("📈 Evolución Histórica de la Probabilidad de Falla de este Equipo")
        st.caption("Gráfico interactivo con todas las fechas evaluadas. Observa los picos de alerta crítica en los días previos a fallas reales.")
        timeline_fig = plot_equipment_timeline(df_equipo_preds, current_date=selected_date)
        st.plotly_chart(timeline_fig, use_container_width=True)

    # 4. Formulario con las variables operativas de esa fecha
    st.markdown("---")
    st.subheader(f"⚙️ Sensores y Telemetría al {selected_date}")
    st.caption("Los valores se han autocompletado con la medición real de esa fecha. Puedes modificarlos para simular qué pasaría si cambias la temperatura o vibración.")

    col_var1, col_var2, col_var3 = st.columns(3)

    with col_var1:
        temp = st.number_input(
            "Temperatura de Proceso (°C)",
            value=float(active_row.get("Temperatura_Proceso", 4.0)),
            step=0.1,
            format="%.2f",
            key=f"temp_{selected_id}_{selected_date}",
        )
        vib = st.number_input(
            "Vibración del Equipo (mm/s)",
            value=float(active_row.get("Vibracion_Equipo", 1.2)),
            step=0.05,
            format="%.3f",
            key=f"vib_{selected_id}_{selected_date}",
        )
        pres = st.number_input(
            "Presión del Sistema (bar)",
            value=float(active_row.get("Presion_Sistema", 2.0)),
            step=0.1,
            format="%.2f",
            key=f"pres_{selected_id}_{selected_date}",
        )

    with col_var2:
        corr = st.number_input(
            "Corriente del Motor (A)",
            value=float(active_row.get("Corriente_Motor", 36.0)),
            step=0.5,
            format="%.2f",
            key=f"corr_{selected_id}_{selected_date}",
        )
        caudal = st.number_input(
            "Caudal de Proceso (L/h)",
            value=float(active_row.get("Caudal_Proceso", 105.0)),
            step=1.0,
            format="%.2f",
            key=f"caud_{selected_id}_{selected_date}",
        )
        nivel = st.number_input(
            "Nivel del Sistema (%)",
            value=float(active_row.get("Nivel_Sistema", 75.0)),
            step=0.5,
            format="%.2f",
            key=f"niv_{selected_id}_{selected_date}",
        )

    with col_var3:
        vel = st.number_input(
            "Velocidad de Accionamiento (RPM)",
            value=float(active_row.get("Velocidad_Accionamiento", 1080.0)),
            step=5.0,
            format="%.1f",
            key=f"vel_{selected_id}_{selected_date}",
        )
        horas = st.number_input(
            "Horas de Operación Acumuladas",
            value=float(active_row.get("Horas_Operacion", 14000.0)),
            step=10.0,
            format="%.1f",
            key=f"hrs_{selected_id}_{selected_date}",
        )
        carga = st.number_input(
            "Carga Electromecánica Estimada",
            value=float(active_row.get("Carga_Electromecanica", (vib * pres))),
            step=1.0,
            format="%.2f",
            key=f"crg_{selected_id}_{selected_date}",
        )

    # Construir vector de características
    feature_row = active_row.copy()
    feature_row["Temperatura_Proceso"] = temp
    feature_row["Vibracion_Equipo"] = vib
    feature_row["Presion_Sistema"] = pres
    feature_row["Corriente_Motor"] = corr
    feature_row["Caudal_Proceso"] = caudal
    feature_row["Nivel_Sistema"] = nivel
    feature_row["Velocidad_Accionamiento"] = vel
    feature_row["Horas_Operacion"] = horas
    feature_row["Carga_Electromecanica"] = carga
    feature_row["Ratio_Presion_Caudal"] = pres / caudal if caudal > 0 else 0.0

    features_dict = {col: float(feature_row[col]) for col in feature_columns}

    # 5. Botón de Inferencia con FastAPI
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 REALIZAR PREDICCIÓN EN ESTA FECHA (VÍA FASTAPI)", type="primary", use_container_width=True):
        with st.spinner("Enviando telemetría al backend FastAPI (POST /predict)..."):
            try:
                api_resp = predict_via_api(
                    api_url=config.api_base_url,
                    equipo=f"EQ-{selected_id} ({active_row.get('Nombre_Equipo')})",
                    features=features_dict,
                )

                risk_info = config.get_risk_level(api_resp["probabilidad_falla_7_dias"])

                st.markdown("### 📊 Diagnóstico Inmediato de FastAPI")
                render_prediction_result_box(api_resp, risk_info)

                # Registrar en el historial
                record = {
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Fecha_Telemetria": selected_date,
                    "Equipo": active_row.get("Nombre_Equipo", f"Equipo {selected_id}"),
                    "ID_Equipo": f"EQ-{selected_id}",
                    "Tipo_Equipo": active_row.get("Tipo_Equipo", "N/A"),
                    "Línea": active_row.get("Línea", "N/A"),
                    "Modelo": f"{api_resp['modelo']} ({api_resp['version_modelo']})",
                    "Probabilidad": api_resp["probabilidad_falla_7_dias"],
                    "Probabilidad (%)": f"{api_resp['probabilidad_porcentaje']:.2f} %",
                    "Riesgo": api_resp["nivel_riesgo"],
                    "Prediccion_Clase": api_resp["prediccion_clase"],
                    "Estado": "Alerta de Falla" if api_resp["prediccion_clase"] == 1 else "Normal",
                }
                save_prediction_to_history(config, record)
                st.success(f"✅ Inferencia completada y registrada en el historial para la fecha {selected_date}.")

            except Exception as e:
                st.error(
                    f"⚠️ **API no disponible:** Verifique que FastAPI esté en ejecución (`uvicorn services.api.main:app --reload`).\n\n"
                    f"*Detalle:* {str(e)}"
                )

    # 6. Tabla completa de todas las observaciones y probabilidades de este equipo
    if not df_equipo_preds.empty:
        st.markdown("---")
        st.subheader(f"📋 Historial Completo de Predicciones del Modelo para {active_row.get('Nombre_Equipo')}")
        st.caption(f"Mostrando todas las {len(df_equipo_preds)} fechas evaluadas en el conjunto de prueba independiente.")

        cols_table = [
            col for col in [
                "Fecha_Observacion",
                "Falla_Real",
                "Prediccion",
                "Probabilidad_Falla_7_Dias_Pct",
                "Criticidad",
                "Línea",
                "Modelo",
            ] if col in df_equipo_preds.columns
        ]
        st.dataframe(
            df_equipo_preds[cols_table].rename(
                columns={
                    "Fecha_Observacion": "Fecha Medición",
                    "Falla_Real": "Falla Real (7D)",
                    "Prediccion": "Predicción (Alerta)",
                    "Probabilidad_Falla_7_Dias_Pct": "Probabilidad de Falla (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
