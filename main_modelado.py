"""Script principal para ejecutar la Fase 7.4: Modelado de Machine Learning."""

import os
import sys
from pathlib import Path

# Configurar ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from services.modeling.main import main

if __name__ == "__main__":
    main()
