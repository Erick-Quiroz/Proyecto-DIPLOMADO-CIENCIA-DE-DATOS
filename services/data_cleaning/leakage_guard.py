"""Módulo para la auditoría formal y prevención de Data Leakage."""

from typing import List, Dict
import pandas as pd


class LeakageGuard:
    """Verificador y documentador de fuga de información y pertinencia de variables."""

    @staticmethod
    def generate_leakage_audit_table() -> pd.DataFrame:
        # Tabla formal de auditoria de data leakage y seleccion de variables
        audit_data = [
            {
                "Variable": "Temperatura_Proceso",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Variable física clave del proceso térmico, disponible en tiempo real.",
            },
            {
                "Variable": "Vibracion_Equipo",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Indicador directo de desbalance mecánico y desgaste de rodamientos.",
            },
            {
                "Variable": "Presion_Sistema",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Mide sobrepresión o caídas en circuitos hidráulicos y de refrigeración.",
            },
            {
                "Variable": "Corriente_Motor",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Refleja sobrecarga eléctrica y resistencia mecánica del accionamiento.",
            },
            {
                "Variable": "Caudal_Proceso",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Monitorea la fluidez y entrega volumétrica en líneas de producción.",
            },
            {
                "Variable": "Nivel_Sistema",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Supervisa niveles de tolvas y depósitos de enfriamiento.",
            },
            {
                "Variable": "Velocidad_Accionamiento",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Registra régimen de rotación y estabilidad de velocidad.",
            },
            {
                "Variable": "Horas_Operacion",
                "Categoria": "Predictora Operativa",
                "Decision": "Utilizada",
                "Motivo": "Variable acumulativa de uso previo al momento de predicción.",
            },
            {
                "Variable": "Tipo_Equipo",
                "Categoria": "Predictora Contextual",
                "Decision": "Utilizada (Codificada)",
                "Motivo": "Aporta contexto funcional sobre el comportamiento de cada familia de máquina.",
            },
            {
                "Variable": "Proceso",
                "Categoria": "Predictora Contextual",
                "Decision": "Utilizada (Codificada)",
                "Motivo": "Contexto del proceso productivo (Congelación, Envasado, etc.).",
            },
            {
                "Variable": "Línea",
                "Categoria": "Predictora Contextual",
                "Decision": "Utilizada (Codificada)",
                "Motivo": "Línea de producción donde opera el equipo.",
            },
            {
                "Variable": "Criticidad",
                "Categoria": "Predictora Contextual",
                "Decision": "Utilizada (Codificada Ordinal)",
                "Motivo": "Jerarquía de criticidad operacional (Baja=0, Media=1, Alta=2).",
            },
            {
                "Variable": "Identificador_Equipo",
                "Categoria": "Identificador",
                "Decision": "Descartada",
                "Motivo": "Identificador único. Su inclusión causaría memorización y sobreajuste.",
            },
            {
                "Variable": "Codigo_Equipo_Origen",
                "Categoria": "Identificador",
                "Decision": "Descartada",
                "Motivo": "Código alfanumérico identificador específico del equipo.",
            },
            {
                "Variable": "Nombre_Equipo",
                "Categoria": "Identificador",
                "Decision": "Descartada",
                "Motivo": "Texto descriptivo que identifica directamente al equipo.",
            },
            {
                "Variable": "Fecha_Observacion",
                "Categoria": "Control Temporal",
                "Decision": "Descartada como predictor",
                "Motivo": "Utilizada exclusivamente para ordenar y dividir cronológicamente los conjuntos.",
            },
            {
                "Variable": "Tiene_Horizonte_Completo_Siete_Dias",
                "Categoria": "Metadato de Control",
                "Decision": "Descartada",
                "Motivo": "Indicador de soporte utilizado para filtrar las 203 filas sin horizonte completo.",
            },
            {
                "Variable": "Conjunto_Temporal",
                "Categoria": "Metadato de Partición",
                "Decision": "Descartada como predictor",
                "Motivo": "Etiqueta original de partición temporal (Train/Valid/Test).",
            },
            {
                "Variable": "Variables de hechos_fallas.csv (Costo, Duracion, Causa)",
                "Categoria": "Información Post-Evento",
                "Decision": "Descartada por Data Leakage",
                "Motivo": "Ocurren con posterioridad a la falla. Usarlas como predictor crearía fuga directa.",
            },
            {
                "Variable": "Falla_En_Los_Siguientes_Siete_Dias",
                "Categoria": "Variable Objetivo (Target)",
                "Decision": "Definida como Target (y)",
                "Motivo": "Etiqueta supervisada a predecir a 7 días de horizonte.",
            },
        ]

        return pd.DataFrame(audit_data)
