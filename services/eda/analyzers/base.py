"""Clase base abstracta y estructuras de datos para los analizadores de EDA."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from services.eda.config import EDAConfig


@dataclass
class AnalysisResult:
    """Resultado estructurado de un analizador específico."""

    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """Clase base abstracta para todos los analizadores del pipeline EDA."""

    def __init__(self, config: EDAConfig):
        self.config = config

    @abstractmethod
    def analyze(self, datasets: Dict[str, pd.DataFrame]) -> AnalysisResult:
        """Ejecuta el análisis sobre los conjuntos de datos proporcionados."""
        pass
