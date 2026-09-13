"""EDA Service Package

Servicio modular e independiente para Análisis Exploratorio de Datos (EDA)
orientado a arquitectura MLOps.
"""

from services.eda.config import EDAConfig
from services.eda.pipeline import EDAPipeline

__all__ = ["EDAConfig", "EDAPipeline"]
