"""Módulo de la interfaz Streamlit para la Carga, Validación y Ejecución del Pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from services.dashboard.config import DashboardConfig
from services.data_cleaning.validator import DataValidator, RAW_DATASETS_SCHEMAS
from services.pipeline_runner import FullPipelineRunner


def render_dataset_upload_card(
    schema_key: str,
    raw_data_dir: Path,
    validator: DataValidator,
):
    """Renderiza el campo de carga y estado para un dataset específico."""
    schema = RAW_DATASETS_SCHEMAS[schema_key]
    dest_path = raw_data_dir / schema.filename

    # Validar el archivo actual en disco si existe
    val_res = validator.validate_file_on_disk(schema.filename)
    status = val_res["status"]

    # Definir colores y etiquetas de estado
    if status == "valid":
        status_text = "Cargado y Válido"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "#10B981"
        badge_color = "#059669"
    elif status == "warning":
        status_text = "Cargado con Errores"
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "#F59E0B"
        badge_color = "#D97706"
    else:
        status_text = "Faltante" if schema.is_mandatory else "No Cargado (Opcional)"
        badge_bg = "rgba(239, 68, 68, 0.15)"
        badge_border = "#EF4444"
        badge_color = "#DC2626"

    # Tarjeta de dataset
    st.markdown(
        f"""
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; margin-bottom: 16px; background-color: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #1E293B;">
                    {schema.label} <code style="font-size: 0.82rem; background: #F1F5F9; color: #475569; padding: 2px 6px; border-radius: 4px;">{schema.filename}</code>
                </div>
                <div style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 700;">
                    {status_text}
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">
                {schema.description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        # File uploader independiente
        uploaded_file = st.file_uploader(
            f"Seleccionar nuevo {schema.filename}:",
            type=["csv"],
            key=f"uploader_{schema_key}",
            help=f"Cargue el archivo CSV correspondiente a {schema.label}. Columnas requeridas: {', '.join(schema.required_columns[:4])}...",
        )

        if uploaded_file is not None:
            # Procesar el archivo subido
            try:
                df_uploaded = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
            except Exception:
                uploaded_file.seek(0)
                df_uploaded = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)

            # Validar en memoria
            validation_memory = validator.validate_single_dataframe(df_uploaded, schema, source_name=uploaded_file.name)

            if validation_memory["errors"]:
                st.warning("El archivo subido presenta observaciones de validación:")
                for err in validation_memory["errors"]:
                    st.markdown(f"- **{err['error_type']}** ({err['column']}): {err['description']}")

            # Guardar el archivo en data/raw/
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            df_uploaded.to_csv(dest_path, index=False, encoding="utf-8-sig")
            st.success(f"Archivo guardado correctamente en data/raw/{schema.filename} ({len(df_uploaded):,d} filas).")
            # Forzar revalidación visual
            st.rerun()

    with col2:
        # Mostrar resumen de estado actual
        if status == "valid":
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #334155; line-height: 1.6; background: #F8FAFC; padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0;">
                    <div><strong>Registros:</strong> {val_res['row_count']:,d} filas</div>
                    <div><strong>Columnas:</strong> {val_res['col_count']} variables</div>
                    {f"<div><strong>Equipos detectados:</strong> {val_res['unique_equipos']}</div>" if val_res['unique_equipos'] > 0 else ""}
                    <div style="color: #059669; font-weight: 600; margin-top: 4px;">Todas las columnas requeridas presentes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status == "warning":
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #991B1B; line-height: 1.5; background: #FEF2F2; padding: 10px 14px; border-radius: 8px; border: 1px solid #FCA5A5;">
                    <div style="font-weight: 700;">Problemas detectados:</div>
                    {"".join([f"<div>• {e['description']}</div>" for e in val_res['errors'][:3]])}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5; background: #F8FAFC; padding: 10px 14px; border-radius: 8px; border: 1px dashed #CBD5E1;">
                    <div><strong>Archivo no detectado.</strong></div>
                    <div style="margin-top: 4px;">Suba el archivo correspondiente o asegúrese de que exista en <code>data/raw/{schema.filename}</code>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='margin: 10px 0 20px 0; border: none; border-top: 1px dashed #E2E8F0;'>", unsafe_allow_html=True)


def render_data_pipeline_page(config: DashboardConfig):
    """Página principal de ingestión de CSVs y orquestación del pipeline."""
    st.markdown("<div class='main-header'>Carga de Datos Crudos y Pipeline de Ciencia de Datos</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Cargue nuevos datasets CSV independientes, valide esquemas en tiempo real y ejecute nuevamente todo el proceso de Ciencia de Datos desde cero.</div>",
        unsafe_allow_html=True,
    )

    # 1. Diagrama del flujo de trabajo
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); color: #F8FAFC; padding: 16px 22px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;">
                Flujo Operativo de Datos
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.86rem; font-weight: 600;">
                <span style="background: #334155; padding: 4px 10px; border-radius: 6px;">1. Cargar CSVs</span>
                <span style="color: #94A3B8;">➔</span>
                <span style="background: #334155; padding: 4px 10px; border-radius: 6px;">2. Validar Esquemas</span>
                <span style="color: #94A3B8;">➔</span>
                <span style="background: #2563EB; color: white; padding: 4px 10px; border-radius: 6px;">3. Limpieza & Split</span>
                <span style="color: #94A3B8;">➔</span>
                <span style="background: #334155; padding: 4px 10px; border-radius: 6px;">4. EDA</span>
                <span style="color: #94A3B8;">➔</span>
                <span style="background: #334155; padding: 4px 10px; border-radius: 6px;">5. Modelado (RF/LR/XGB)</span>
                <span style="color: #94A3B8;">➔</span>
                <span style="background: #10B981; color: white; padding: 4px 10px; border-radius: 6px;">6. Métricas & Resultados</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_data_dir = getattr(config, "raw_data_dir", None) or (config.base_dir / "data" / "raw")
    validator = DataValidator(raw_data_dir)

    # 2. Sección de carga de cada CSV
    st.subheader("1. Carga Independiente de Datasets Crudos")
    st.caption("Cada dataset cuenta con su propio campo de carga dedicado. No se mezclarán ni asumirán nombres genéricos.")

    for schema_key in RAW_DATASETS_SCHEMAS.keys():
        render_dataset_upload_card(
            schema_key=schema_key,
            raw_data_dir=raw_data_dir,
            validator=validator,
        )

    # 3. Resumen y Validación Obligatoria Previa
    st.subheader("2. Validación de Estado de Datasets")
    val_summary = validator.validate_all_mandatory_datasets()
    can_execute = val_summary["can_execute_pipeline"]

    col_st1, col_st2 = st.columns([1.2, 1])

    with col_st1:
        st.markdown("**Estado actual de los datasets requeridos:**")
        for fn, res in val_summary["datasets"].items():
            if res["status"] == "valid":
                st.markdown(f"**`{fn}`** — Válido ({res['row_count']:,d} registros, {res['col_count']} columnas)")
            elif res["status"] == "warning":
                st.markdown(f"**`{fn}`** — Con observaciones ({len(res['errors'])} inconsistencias detectadas)")
            else:
                st.markdown(f"**`{fn}`** — **FALTANTE OBLIGATORIO**")

    with col_st2:
        if can_execute:
            st.markdown(
                """
                <div style="background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 14px 18px; border-radius: 8px; color: #065F46;">
                    <div style="font-weight: 700; font-size: 0.95rem;">Todos los datasets obligatorios están listos</div>
                    <div style="font-size: 0.82rem; margin-top: 4px;">Esquemas y columnas validados. Puede iniciar la ejecución completa del pipeline.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            faltantes_str = ", ".join(val_summary["missing_datasets"]) if val_summary["missing_datasets"] else "Ninguno"
            con_error_str = ", ".join(val_summary["datasets_with_errors"]) if val_summary["datasets_with_errors"] else "Ninguno"
            st.markdown(
                f"""
                <div style="background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 14px 18px; border-radius: 8px; color: #991B1B;">
                    <div style="font-weight: 700; font-size: 0.95rem;">No se puede ejecutar el pipeline</div>
                    <div style="font-size: 0.82rem; margin-top: 4px;">
                        <strong>Faltantes:</strong> {faltantes_str}<br>
                        <strong>Con errores:</strong> {con_error_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Detalle de Errores si existen
    if val_summary["all_errors"]:
        with st.expander("Ver Detalle de Inconsistencias de Validación", expanded=not can_execute):
            for err in val_summary["all_errors"]:
                st.error(
                    f"**Dataset:** `{err['dataset']}` | **Columna:** `{err['column']}` | **Tipo:** {err['error_type']}\n\n"
                    f"*{err['description']}* (Filas afectadas: {err.get('affected_rows', 0):,d})"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Botón de Ejecución del Pipeline
    st.subheader("3. Ejecución Completa del Pipeline de Ciencia de Datos")
    st.caption("Re-entrenará los modelos, recalculará el EDA, dividirá los conjuntos y regenerará todas las métricas exclusivamente con los nuevos datos.")

    btn_col1, btn_col2 = st.columns([1, 1.5])

    with btn_col1:
        run_btn = st.button(
            "EJECUTAR PIPELINE COMPLETO",
            type="primary",
            use_container_width=True,
            disabled=not can_execute,
            help="Habilitado únicamente cuando todos los CSVs obligatorios han sido validados correctamente.",
        )

    if run_btn and can_execute:
        progress_bar = st.progress(0, text="Iniciando ejecución...")
        status_box = st.empty()
        log_container = st.container()

        def update_progress(step: int, total: int, message: str):
            pct = int((step / total) * 100)
            progress_bar.progress(pct, text=f"Paso {step}/{total}: {message}")
            status_box.info(f"**Ejecutando:** {message}")

        try:
            with st.spinner("Procesando pipeline de Ciencia de Datos..."):
                runner = FullPipelineRunner(base_dir=config.base_dir)
                results = runner.run_pipeline(progress_callback=update_progress)

            # Limpiar caché de Streamlit para recargar todo de forma inmediata
            st.cache_data.clear()

            progress_bar.progress(100, text="Pipeline finalizado con éxito.")
            status_box.success(f"Pipeline de Ciencia de Datos completado con éxito en {results['elapsed_seconds']} segundos.")

            # Mostrar resumen de resultados recalculados
            st.markdown("---")
            st.subheader("Resultados Recalculados Desde Cero con los Nuevos Datos")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Total Observaciones", f"{results['filas_validas']:,d}")
            with kpi2:
                st.metric("Equipos Únicos", f"{results['equipos_unicos']}")
            with kpi3:
                st.metric("Variables Predictoras", f"{results['total_features']}")
            with kpi4:
                st.metric("Modelo Seleccionado", f"{results['best_model_name']}")

            # Tabla comparativa de modelos recalculada
            if results.get("tabla_comparacion") is not None and not results["tabla_comparacion"].empty:
                st.markdown("**Comparativa de Modelos Recalculada en Conjunto de Prueba:**")
                st.dataframe(
                    results["tabla_comparacion"].style.format({
                        "Accuracy": "{:.2%}",
                        "Precision": "{:.2%}",
                        "Recall": "{:.2%}",
                        "F1-Score": "{:.2%}",
                        "ROC-AUC": "{:.2%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            st.success("Todos los resultados, predicciones, métricas y gráficos del panel han sido recalculados automáticamente.")

            # Acciones posteriores
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("Ir al Panel General", use_container_width=True):
                    st.session_state["current_page"] = "Panel General"
                    st.rerun()
            with col_act2:
                if st.button("Ir a Análisis de Modelos", use_container_width=True):
                    st.session_state["current_page"] = "Análisis de Modelos"
                    st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_box.error(f"Error durante la ejecución del pipeline: {str(e)}")
            st.exception(e)
