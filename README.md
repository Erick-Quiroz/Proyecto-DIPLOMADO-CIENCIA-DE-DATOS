# Proyecto Final: Predicción de Fallas en Equipos Industriales (Área de Helados)

## 📌 Descripción del Proyecto

Este repositorio contiene la solución técnica integral para el proyecto de Ciencia de Datos y Machine Learning:

> **Título:** *“Sistema Predictivo para la Estimación de la Probabilidad de Falla en Equipos Industriales del Área de Helados mediante Técnicas de Machine Learning”*  
> **Problema:** *“¿Cómo estimar la probabilidad de falla de los equipos industriales del Área de Helados durante los siguientes siete días, a partir del análisis de sus variables de operación y registros históricos de fallas, mediante técnicas de Ciencia de Datos?”*

El sistema está diseñado bajo una **arquitectura desacoplada y profesional de microservicios MLOps**, garantizando reproducibilidad estricta, prevención total de fuga de información (*Data Leakage*) y partición temporal cronológica.


### 1. Configurar el Entorno Virtual (`.venv`)
```bash
# Crear y aprovisionar el entorno unificado automáticamente
python scripts/setup_environment.py --mode unified
```

### 2. Activar el Entorno Virtual

- **En Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **En Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **En Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

---

##  Comandos de Ejecución y Salidas

| Flujo / Servicio | Comando | Salidas Principales |
| :--- | :--- | :--- |
| **Pipeline Completo (Fase 7.3)** | `python main_preparacion.py` | `data/processed/dataset_modelado.csv`<br>`data/processed/X_train.csv`, `y_train.csv`<br>`data/processed/X_valid.csv`, `y_valid.csv`<br>`data/processed/X_test.csv`, `y_test.csv` |
| **Limpieza de Datos (`data_cleaning`)** | `python -m services.data_cleaning.main` | `data/processed/observaciones_limpias.csv`<br>`tablas_preparacion/auditoria_*.csv`<br>`tablas_preparacion/matriz_data_leakage.csv` |
| **Ingeniería de Características (`feature_engineering`)** | `python -m services.feature_engineering.main` | Matrices `X_` / `y_` para Train, Valid y Test<br>`tablas_preparacion/resumen_*.csv`<br>`figuras_preparacion/*.png` |
| **Análisis Exploratorio (`eda`)** | `python -m services.eda.main` | `reports/figures/eda/*.png`<br>`reports/tables/eda/*.csv` |

---

##  Metodología y Preparación de Datos (Fase 7.3)

```
[Datos Crudos: observaciones_diarias] (17,719 filas)
                  │
                  ▼
[Auditoría y Filtro de Horizonte] ──► Exclusión de 203 filas sin ventana 7D (Total: 17,516 filas)
                  │
                  ▼
[Ingeniería Temporal] ──────────────► Medias 7D, Std 7D, Deltas 1D y Ratios de Esfuerzo (sin lookahead)
                  │
                  ▼
[Codificación Categórica] ──────────► One-Hot y Ordinal ajustados estrictamente en TRAIN
                  │
                  ▼
[División Cronológica]
   ├── Entrenamiento (Train) : 12,412 registros (70.86%) | 1,323 fallas (10.66%) [2025-01-01 a 2026-03-04]
   ├── Validación (Valid)    :  2,668 registros (15.23%) |   313 fallas (11.73%) [2026-03-05 a 2026-06-04]
   └── Prueba (Test)         :  2,436 registros (13.91%) |   310 fallas (12.73%) [2026-06-05 a 2026-08-27]
```

### Prevención de Data Leakage
- **Variables descartadas como predictores:** `Identificador_Equipo`, `Codigo_Equipo_Origen`, `Nombre_Equipo` (evitan memorización y sobreajuste).
- **Información post-evento descartada:** Variables de costos, causas y paradas de `hechos_fallas.csv`.
- **Ajuste de transformadores:** Todos los codificadores y escaladores se ajustan exclusivamente sobre el conjunto de Entrenamiento.

---

## Estructura del Proyecto

```text
Proyecto-DIPLOMADO-CIENCIA-DE-DATOS/
├── .venv/                      # Entorno virtual aislado (ignorado en git)
├── data/                       # Almacenamiento organizado de datos (ignorado en git)
│   ├── raw/                    # Datos fuentes originales e inmutables
│   ├── interim/                # Datos intermedios transformados
│   └── processed/              # Datasets finales listos para modelado (X_train, y_train, etc.)
├── services/                   # Microservicios modulares desacoplados (MLOps)
│   ├── data_cleaning/          # Validación, auditoría de atípicos/nulos y filtro de horizonte
│   ├── feature_engineering/    # Cálculo de features temporales, codificación y partición cronológica
│   ├── training/               # Entrenamiento y evaluación de algoritmos de Machine Learning
│   ├── eda/                    # Análisis exploratorio automatizado de datos
│   ├── serving/                # Inferencia y API REST para predicción en producción
│   └── monitoring/             # Monitoreo de deriva de datos (Data Drift) y métricas
├── tablas_preparacion/         # Tablas CSV de evidencia de la Fase 7.3
├── figuras_preparacion/        # Gráficos PNG de control temporal y distribución de clases
├── notebooks/                  # Cuadernos Jupyter para experimentación y EDA interactivo
├── reports/                    # Reportes ejecutivos, figuras y tablas de análisis
├── scripts/                    # Scripts de soporte y gestión de entornos virtuales
├── src/                        # Código base modular empaquetado (pip install -e .)
├── models/                     # Modelos serializados entrenados (.pkl, .joblib)
├── docs/                       # Documentación técnica
└── references/                 # Diccionarios de datos y manuales de planta
```

---

## Tecnologías Utilizadas

- **Plantilla Base:** [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- **Arquitectura:** Microservicios Modulares desacoplados (MLOps)
- **Lenguaje:** Python 3.12
- **Procesamiento de Datos:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-Learn
- **Visualización:** Matplotlib, Seaborn
- **Gestión de Entornos:** Virtualenv, Pip, SetupTools
- **Control de Versiones:** Git, GitHub

---

## Autor

- **Erick Quiroz** - [GitHub Profile](https://github.com/Erick-Quiroz)
- **Diplomado en Ciencia de Datos Aplicada**