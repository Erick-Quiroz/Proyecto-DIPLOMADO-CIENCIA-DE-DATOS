"""Script principal reproducible para la Fase 7.5: Evaluación y Resultados (CRISP-DM)."""

import os
import sys
from pathlib import Path

# Configurar ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from services.evaluation.main import main

if __name__ == "__main__":
    main()
