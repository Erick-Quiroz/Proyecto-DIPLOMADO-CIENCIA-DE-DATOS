"""Componentes visuales, tarjetas KPI, estado de API y gráficos interactivos con Plotly."""

from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.dashboard.config import DashboardConfig


def apply_custom_styles():
    """Aplica estilos CSS para dar una apariencia profesional uniforme inspirada en Bootstrap 5 y AdminLTE 4."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Tipografía Global y Variables de Color */
        :root {
            --admin-primary: #2563EB;
            --admin-primary-dark: #1D4ED8;
            --admin-sidebar-bg: #0F172A;
            --admin-sidebar-card: #1E293B;
            --admin-card-bg: #FFFFFF;
            --admin-border: #E2E8F0;
            --admin-text-main: #0F172A;
            --admin-text-muted: #64748B;
        }

        html, body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Cabeceras de Página Principales */
        .main-header {
            font-size: 2.15rem;
            font-weight: 800;
            color: #000000 !important; /* Negro en Light Mode por defecto */
            margin-top: 0.2rem;
            margin-bottom: 6px;
            letter-spacing: -0.025em;
            display: flex;
            align-items: center;
            gap: 12px;
            line-height: 1.25;
        }

        .sub-header {
            font-size: 1.02rem;
            color: #4B5563 !important;
            margin-bottom: 1.5rem;
            font-weight: 500;
            line-height: 1.55;
        }

        /* Variable de texto nativa de Streamlit */
        @supports (color: var(--text-color)) {
            .main-header {
                color: var(--text-color, #000000) !important;
            }
        }

        /* Soporte para modo oscuro vía Media Query */
        @media (prefers-color-scheme: dark) {
            .main-header {
                color: #FFFFFF !important; /* Blanco puro en Modo Oscuro */
            }
            .sub-header {
                color: #D1D5DB !important;
            }
        }

        /* Forzar modo oscuro cuando Streamlit o el contenedor activa dark theme */
        [data-theme="dark"] .main-header,
        .stApp[data-theme="dark"] .main-header,
        [data-testid="stAppViewContainer"][data-theme="dark"] .main-header,
        .stApp[class*="dark"] .main-header,
        .dark .main-header {
            color: #FFFFFF !important; /* Blanco puro en Dark Mode */
        }

        [data-theme="dark"] .sub-header,
        .stApp[data-theme="dark"] .sub-header,
        [data-testid="stAppViewContainer"][data-theme="dark"] .sub-header,
        .stApp[class*="dark"] .sub-header,
        .dark .sub-header {
            color: #D1D5DB !important;
        }

        /* Forzar modo claro cuando Streamlit o el contenedor activa light theme */
        [data-theme="light"] .main-header,
        .stApp[data-theme="light"] .main-header,
        [data-testid="stAppViewContainer"][data-theme="light"] .main-header,
        .stApp[class*="light"] .main-header,
        .light .main-header {
            color: #000000 !important; /* Negro puro en Light Mode */
        }

        [data-theme="light"] .sub-header,
        .stApp[data-theme="light"] .sub-header,
        [data-testid="stAppViewContainer"][data-theme="light"] .sub-header,
        .stApp[class*="light"] .sub-header,
        .light .sub-header {
            color: #4B5563 !important;
        }

        /* Sidebar Styling (AdminLTE Dark Theme) */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B;
        }

        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }

        /* Brand Logo Area: Ciencia de Datos / v1.0.0 */
        .admin-brand-box {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 4px 18px 4px;
            border-bottom: 1px solid #1E293B;
            margin-bottom: 20px;
        }

        .admin-brand-icon {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.35);
        }

        .admin-brand-text {
            display: flex;
            flex-direction: column;
        }

        .admin-brand-name {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            line-height: 1.2;
        }

        .admin-brand-version {
            font-size: 0.75rem;
            color: #60A5FA;
            font-weight: 700;
            letter-spacing: 0.05em;
        }

        /* Sidebar Section Headers */
        .sidebar-section-title {
            font-size: 0.70rem;
            font-weight: 700;
            color: #64748B;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 18px 4px 10px 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Botones de Navegación del Sidebar 100% UNIFORMES (Estilo AdminLTE Nav-Item) */
        [data-testid="stSidebar"] .stButton {
            width: 100% !important;
            margin-bottom: 6px !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100% !important;
            height: 46px !important;
            min-height: 46px !important;
            max-height: 46px !important;
            border-radius: 8px !important;
            font-size: 0.92rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding: 0 16px !important;
            box-sizing: border-box !important;
            transition: all 0.2s ease-in-out !important;
        }

        /* Botón Activo (Primary) */
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            border: 1px solid #3B82F6 !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
        }

        /* Botón Inactivo (Secondary) */
        [data-testid="stSidebar"] .stButton > button[kind="secondary"],
        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {
            background: #1E293B !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #CBD5E1 !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            background: #334155 !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
            color: #FFFFFF !important;
            transform: translateX(3px) !important;
        }

        /* Botones Globales en Contenido Principal */
        .main .stButton > button {
            width: 100% !important;
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }

        /* Telemetry Status Card (AdminLTE Sidebar Widget) */
        .admin-status-box {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            margin-top: 14px;
            margin-bottom: 14px;
            box-sizing: border-box;
            width: 100%;
        }

        /* AdminLTE "Small Box" / KPI Cards */
        .admin-small-box {
            position: relative;
            display: block;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease-in-out;
            overflow: hidden;
            height: 125px;
            box-sizing: border-box;
        }

        .admin-small-box:hover {
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
            border-color: #CBD5E1;
        }

        .small-box-inner {
            position: relative;
            z-index: 2;
        }

        .small-box-number {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 6px;
        }

        .small-box-title {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 4px;
        }

        .small-box-subtitle {
            font-size: 0.78rem;
            color: #94A3B8;
            font-weight: 500;
        }

        .small-box-icon {
            position: absolute;
            top: 12px;
            right: 16px;
            font-size: 2.6rem;
            opacity: 0.15;
            z-index: 1;
            pointer-events: none;
            transition: transform 0.3s ease;
        }

        .admin-small-box:hover .small-box-icon {
            transform: scale(1.1) rotate(-5deg);
            opacity: 0.25;
        }

        /* Info box */
        .custom-info-box {
            background-color: #F8FAFC;
            border-left: 4px solid #2563EB;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin: 14px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            color: #334155;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_api_status_badge(is_connected: bool, active_model: str, api_url: str):
    """Renderiza el estado de conexión del backend FastAPI en el Sidebar con estilo AdminLTE."""
    if is_connected:
        st.markdown(
            f"""
            <div class="admin-status-box" style="border-left: 4px solid #10B981;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.74rem; font-weight: 700; color: #10B981; display: flex; align-items: center; gap: 4px;">
                        ● BACKEND ONLINE
                    </span>
                    <span style="font-size: 0.68rem; color: #94A3B8; font-family: monospace;">PORT 8000</span>
                </div>
                <div style="font-size: 0.74rem; color: #CBD5E1; line-height: 1.4;">
                    <div><strong>Host:</strong> <code style="color: #67E8F9; background: rgba(15,23,42,0.8); padding: 1px 4px; border-radius: 4px;">{api_url}</code></div>
                    <div style="margin-top: 2px;"><strong>Activo:</strong> <code style="color: #A7F3D0; background: rgba(15,23,42,0.8); padding: 1px 4px; border-radius: 4px;">{active_model}</code></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="admin-status-box" style="border-left: 4px solid #EF4444;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.74rem; font-weight: 700; color: #EF4444; display: flex; align-items: center; gap: 4px;">
                        ● API OFFLINE
                    </span>
                    <span style="font-size: 0.68rem; color: #FCA5A5; font-family: monospace;">ERROR</span>
                </div>
                <div style="font-size: 0.72rem; color: #F87171; line-height: 1.3;">
                    Ejecutar backend:<br>
                    <code style="color: #FECACA; background: rgba(15,23,42,0.8); padding: 1px 4px; border-radius: 4px;">uvicorn services.api.main:app --reload</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_cards(
    total_equipos: int,
    equipos_bajo: int,
    equipos_medio: int,
    equipos_alto: int,
):
    """Renderiza tarjetas KPI estilo Small-Box de AdminLTE para el estado de planta."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="admin-small-box" style="border-top: 4px solid #2563EB;">
                <div class="small-box-inner">
                    <div class="small-box-title">Equipos Monitoreados</div>
                    <div class="small-box-number" style="color: #1E3A8A;">{total_equipos}</div>
                    <div class="small-box-subtitle">Área de Helados (24/7)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="admin-small-box" style="border-top: 4px solid #10B981;">
                <div class="small-box-inner">
                    <div class="small-box-title" style="color: #059669;">Operación Normal</div>
                    <div class="small-box-number" style="color: #059669;">{equipos_bajo:,d}</div>
                    <div class="small-box-subtitle">Riesgo Bajo (&lt; 30%)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="admin-small-box" style="border-top: 4px solid #F59E0B;">
                <div class="small-box-inner">
                    <div class="small-box-title" style="color: #D97706;">En Observación</div>
                    <div class="small-box-number" style="color: #D97706;">{equipos_medio:,d}</div>
                    <div class="small-box-subtitle">Riesgo Medio (30% - 59%)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="admin-small-box" style="border-top: 4px solid #EF4444;">
                <div class="small-box-inner">
                    <div class="small-box-title" style="color: #DC2626;">Alertas Críticas</div>
                    <div class="small-box-number" style="color: #DC2626;">{equipos_alto:,d}</div>
                    <div class="small-box-subtitle">Riesgo Falla 7D (&ge; 60%)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_result_box(api_response: Dict[str, Any], risk_info: Dict[str, Any]):
    """Renderiza la tarjeta principal con los resultados de inferencia recibidos de FastAPI."""
    prob_pct = api_response["probabilidad_porcentaje"]
    color = risk_info["color"]
    label = api_response["nivel_riesgo"]
    model_name = api_response["modelo"]
    model_ver = api_response["version_modelo"]
    rec = api_response["recomendacion"]

    # Gauge visual con Plotly
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": " %", "font": {"size": 44, "color": color, "weight": "bold"}},
            title={"text": "PROBABILIDAD DE FALLA EN 7 DÍAS", "font": {"size": 15, "color": "#374151"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9CA3AF"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#F3F4F6",
                "steps": [
                    {"range": [0, 29], "color": "#D4EDDA"},
                    {"range": [30, 59], "color": "#FFF3CD"},
                    {"range": [60, 79], "color": "#FFE8D6"},
                    {"range": [80, 100], "color": "#F8D7DA"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": prob_pct,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div style="background-color: {risk_info['bg_color']}; border-left: 6px solid {color}; padding: 20px; border-radius: 8px; margin-top: 15px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #4B5563; text-transform: uppercase;">Nivel de Riesgo Operativo</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {color}; margin: 5px 0;">
                    RIESGO {label.upper()}
                </div>
                <div style="font-size: 0.95rem; color: #1F2937; margin-top: 8px; line-height: 1.4;">
                    <strong>Recomendación:</strong> {rec}
                </div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 10px;">
                    <strong>Modelo Servido por FastAPI:</strong> {model_name} (<code>{model_ver}</code>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_equipment_timeline(df_equipment: pd.DataFrame, current_date: str = None) -> go.Figure:
    """Genera una línea de tiempo interactiva con la evolución de la probabilidad de falla del equipo."""
    if df_equipment.empty:
        return go.Figure()

    df_sorted = df_equipment.sort_values(by="Fecha_Observacion").copy()
    prob_col = "Probabilidad_Falla_7_Dias_Pct" if "Probabilidad_Falla_7_Dias_Pct" in df_sorted.columns else "Prob_Falla_Random Forest (%)"

    fig = go.Figure()

    # Bandas de riesgo de fondo
    fig.add_hrect(y0=0, y1=29, fillcolor="#28a745", opacity=0.08, line_width=0, annotation_text="Bajo (<30%)", annotation_position="top left")
    fig.add_hrect(y0=30, y1=59, fillcolor="#ffc107", opacity=0.08, line_width=0, annotation_text="Medio (30-59%)", annotation_position="top left")
    fig.add_hrect(y0=60, y1=79, fillcolor="#fd7e14", opacity=0.08, line_width=0, annotation_text="Alto (60-79%)", annotation_position="top left")
    fig.add_hrect(y0=80, y1=100, fillcolor="#dc3545", opacity=0.08, line_width=0, annotation_text="Crítico (≥80%)", annotation_position="top left")

    # Traza principal de probabilidad
    fig.add_trace(
        go.Scatter(
            x=df_sorted["Fecha_Observacion"],
            y=df_sorted[prob_col],
            mode="lines+markers",
            name="Probabilidad de Falla 7D (%)",
            line=dict(color="#1E3A8A", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>Fecha:</b> %{x}<br><b>Probabilidad:</b> %{y:.2f}%<extra></extra>",
        )
    )

    # Marcar eventos de falla real
    if "Falla_Real" in df_sorted.columns:
        df_fallas = df_sorted[df_sorted["Falla_Real"] == 1]
        if not df_fallas.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_fallas["Fecha_Observacion"],
                    y=df_fallas[prob_col],
                    mode="markers",
                    name="🚨 Falla Real Confirmada",
                    marker=dict(color="#DC2626", size=10, symbol="x", line=dict(width=2, color="#7F1D1D")),
                    hovertemplate="<b>🚨 FALLA CONFIRMADA</b><br>Fecha: %{x}<br>Probabilidad: %{y:.2f}%<extra></extra>",
                )
            )

    # Línea vertical para la fecha seleccionada
    if current_date and current_date in df_sorted["Fecha_Observacion"].values:
        fig.add_vline(
            x=current_date,
            line_dash="dash",
            line_color="#7C3AED",
            annotation_text=f"Fecha Seleccionada: {current_date}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"📈 Evolución Temporal de Probabilidad de Falla ({df_sorted['Nombre_Equipo'].iloc[0]})",
        xaxis_title="Fecha de Observación",
        yaxis_title="Probabilidad de Falla (%)",
        yaxis=dict(range=[-2, 105]),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_probability_by_equipment(df_data: pd.DataFrame) -> go.Figure:
    """Genera gráfico de barras ordenado con la máxima probabilidad de falla por equipo."""
    if df_data.empty or "Nombre_Equipo" not in df_data.columns:
        return go.Figure()

    prob_col = (
        "Probabilidad_Falla_7_Dias_Pct"
        if "Probabilidad_Falla_7_Dias_Pct" in df_data.columns
        else "Prob_Falla_Random Forest (%)"
    )

    if prob_col not in df_data.columns:
        return go.Figure()

    df_grouped = (
        df_data.groupby(["Identificador_Equipo", "Nombre_Equipo"])[prob_col]
        .max()
        .reset_index()
        .sort_values(by=prob_col, ascending=True)
    )

    colors = []
    for p in df_grouped[prob_col]:
        if p >= 80:
            colors.append("#dc3545")
        elif p >= 60:
            colors.append("#fd7e14")
        elif p >= 30:
            colors.append("#ffc107")
        else:
            colors.append("#28a745")

    fig = go.Figure(
        go.Bar(
            x=df_grouped[prob_col],
            y=df_grouped["Nombre_Equipo"],
            orientation="h",
            marker_color=colors,
            text=df_grouped[prob_col].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Máxima Probabilidad de Falla por Equipo en Período Evaluado",
        xaxis_title="Probabilidad de Falla (%)",
        yaxis_title="Equipo Industrial",
        height=420,
        margin=dict(l=20, r=40, t=50, b=30),
        xaxis=dict(range=[0, 110]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_risk_distribution_pie(df_data: pd.DataFrame, config: DashboardConfig) -> go.Figure:
    """Genera gráfico de dona con la distribución de niveles de riesgo."""
    prob_col = (
        "Probabilidad_Falla_7_Dias_Pct"
        if "Probabilidad_Falla_7_Dias_Pct" in df_data.columns
        else "Prob_Falla_Random Forest (%)"
    )

    if df_data.empty or prob_col not in df_data.columns:
        return go.Figure()

    levels = []
    for val in df_data[prob_col]:
        prob = val / 100.0 if val > 1.0 else val
        risk_info = config.get_risk_level(prob)
        levels.append(risk_info["label"])

    df_levels = pd.Series(levels).value_counts().reset_index()
    df_levels.columns = ["Nivel", "Cantidad"]

    color_map = {
        "Bajo": "#28a745",
        "Medio": "#ffc107",
        "Alto": "#fd7e14",
        "Crítico": "#dc3545",
    }

    fig = px.pie(
        df_levels,
        names="Nivel",
        values="Cantidad",
        color="Nivel",
        color_discrete_map=color_map,
        hole=0.45,
        title="Distribución Total de Observaciones por Nivel de Riesgo",
    )
    fig.update_traces(textinfo="percent+label", textfont_size=12)
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig
