"""Módulo de autenticación y control de acceso para el Dashboard de Streamlit."""

import os
import streamlit as st
from services.dashboard.config import DashboardConfig


def check_dashboard_auth(config: DashboardConfig) -> bool:
    """Verifica si la sesión actual está autenticada.
    
    Si no lo está, renderiza una interfaz de inicio de sesión moderna y detiene la ejecución.
    Retorna True si el usuario está autenticado, False en caso contrario.
    """
    # Inicializar estado en session_state si no existe
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "logged_user" not in st.session_state:
        st.session_state["logged_user"] = None

    if st.session_state["authenticated"]:
        return True

    # Ocultar barra lateral en pantalla de Login
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        .login-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 32px 28px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            margin-top: 40px;
        }
        .login-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #0F172A;
            text-align: center;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }
        .login-subtitle {
            font-size: 0.88rem;
            color: #64748B;
            text-align: center;
            margin-bottom: 24px;
        }
        .login-badge {
            background: #EFF6FF;
            color: #2563EB;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 12px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.3, 1])

    with center_col:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 20px;">
                <span class="login-badge">SEGURIDAD Y CONTROL DE ACCESO</span>
                <div class="login-title">GELATO PREDICT 4.0</div>
                <div class="login-subtitle">Sistema de Mantenimiento Predictivo — Planta de Helados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### Iniciar Sesión")
            username_input = st.text_input(
                "Usuario",
                placeholder="Ingrese su nombre de usuario",
                key="input_auth_user",
            )
            password_input = st.text_input(
                "Contraseña",
                type="password",
                placeholder="••••••••",
                key="input_auth_pass",
            )

            submit_btn = st.form_submit_button(
                "Ingresar al Sistema",
                use_container_width=True,
                type="primary",
            )

            if submit_btn:
                expected_user = config.dashboard_user.strip()
                expected_pass = config.dashboard_password.strip()

                if username_input.strip() == expected_user and password_input.strip() == expected_pass:
                    st.session_state["authenticated"] = True
                    st.session_state["logged_user"] = username_input.strip()
                    st.success("✓ Credenciales válidas. Accediendo al sistema...")
                    st.rerun()
                else:
                    st.error("Acceso denegado: Usuario o contraseña incorrectos.")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 18px; font-size: 0.75rem; color: #94A3B8;">
                Acceso restringido a personal autorizado de planta y mantenimiento.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return False


def render_sidebar_user_profile(config: DashboardConfig):
    """Renderiza el perfil del usuario autenticado y el botón para cerrar sesión en la barra lateral."""
    user_name = st.session_state.get("logged_user", config.dashboard_user)

    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Operador</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #FFFFFF;">👤 {user_name}</div>
                </div>
                <span style="font-size: 0.68rem; background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; padding: 2px 7px; border-radius: 10px; font-weight: 600;">Activo</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Cerrar Sesión", key="btn_logout_sidebar", use_container_width=True, type="secondary"):
        st.session_state["authenticated"] = False
        st.session_state["logged_user"] = None
        st.rerun()
