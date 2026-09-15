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

---

## Fase 7.6: Despliegue (FastAPI + Streamlit)

El sistema cuenta con una arquitectura de despliegue desacoplada lista para producción:

```
                    USUARIO (Operador / Data Scientist)
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ STREAMLIT (Port 8501)│  → Frontend Web
                         │ - Inicio             │
                         │ - Predicción en vivo │
                         │ - Historial          │
                         │ - Análisis           │
                         │ - Laboratorio        │
                         └──────────┬───────────┘
                                    │  HTTP / JSON
                                    ▼
                         ┌──────────────────────┐
                         │  FASTAPI (Port 8000) │  → Backend API REST
                         │  /health             │
                         │  /models             │
                         │  /predict            │
                         │  /models/set-active  │
                         └──────────┬───────────┘
                                    │  predict_proba()
                                    ▼
                         ┌──────────────────────┐
                         │   MODELO ML ACTIVO   │  → Random Forest (7.5)
                         │   (o Laboratorio)    │
                         └──────────────────────┘
```

### Ejecución del Sistema

Para levantar el sistema completo se ejecutan dos procesos concurrentes:

```bash
# Terminal 1: Iniciar Backend API REST (FastAPI)
uvicorn services.api.main:app --reload --port 8000

# Terminal 2: Iniciar Frontend Web (Streamlit)
streamlit run services/dashboard/app.py
```

* **Frontend Streamlit:** [http://localhost:8501](http://localhost:8501)
* **Backend API REST:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Documentación Interactiva (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoints Principales de la API

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Comprobación de salud del servicio y modelo activo |
| `GET` | `/models` | Lista de modelos serializados (producción y laboratorio) |
| `GET` | `/models/active` | Consulta el modelo actualmente activo |
| `POST` | `/predict` | Inferencia de probabilidad continua en 7 días y nivel de riesgo con validación de 44 variables |
| `POST` | `/models/set-active` | Promueve dinámicamente un modelo nuevo al servicio de producción |

---

## Despliegue e Infraestructura MLOps (Docker, MinIO, DVC, FastAPI, Streamlit)

El sistema implementa una arquitectura reproducible y modular de microservicios contenerizados y versionamiento de datos/modelos:

```
                    ┌───────────────────────────┐
                    │     USUARIO / OPERARIO    │
                    └─────────────┬─────────────┘
                                  │ HTTP (:8501)
                                  ▼
                    ┌───────────────────────────┐
                    │    STREAMLIT DASHBOARD    │
                    │   (docker/dashboard)      │
                    └─────────────┬─────────────┘
                                  │ HTTP (:8000) [API_URL=http://api:8000]
                                  ▼
                    ┌───────────────────────────┐
                    │     FASTAPI INFERENCE     │
                    │       (docker/api)        │
                    └─────────────┬─────────────┘
                                  │ predict_proba()
                                  ▼
                    ┌───────────────────────────┐
                    │    RANDOM FOREST MODEL    │
                    │ modelo_random_forest.joblib│
                    └───────────────────────────┘

        VERSIONAMIENTO Y ALMACENAMIENTO DE DATOS Y MODELOS

                    ┌───────────────────────────┐
                    │            GIT            │
                    │ (Metadatos, código, .dvc) │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │            DVC            │
                    │  (data.dvc, models.dvc)   │
                    └─────────────┬─────────────┘
                                  │ Protocolo S3
                                  ▼
                    ┌───────────────────────────┐
                    │       MINIO STORAGE       │
                    │       (minio:9000)        │
                    │    Bucket: dvc-storage    │
                    └───────────────────────────┘
```

### 1. Requisitos Previos

- **Git** >= 2.30
- **Python** 3.12+ (con gestor `pip` y entorno virtual)
- **Docker** >= 24.0 y **Docker Compose** >= 2.20
- **DVC con soporte S3** (`dvc[s3]`)

### 2. Instalación de DVC con Soporte S3

Para gestionar el versionamiento de datos y modelos vinculados a MinIO S3:

```bash
python -m pip install "dvc[s3]"
dvc --version
```

### 3. Configuración del Archivo de Variables de Entorno (`.env`)

Copia la plantilla de configuración `.env.example` para crear tu archivo local `.env`:

```bash
cp .env.example .env
```

Configura los valores correspondientes en `.env` (las credenciales nunca se suben a Git ni se escriben en Dockerfiles):

```env
# MinIO S3 Storage
MINIO_ROOT_USER=admin_helados
MINIO_ROOT_PASSWORD=<TU_CONTRASENA_SEGURA>
MINIO_BUCKET=dvc-storage
MINIO_ENDPOINT=http://minio:9000
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

# DVC S3 Credentials
AWS_ACCESS_KEY_ID=admin_helados
AWS_SECRET_ACCESS_KEY=<TU_CONTRASENA_SEGURA>
AWS_DEFAULT_REGION=us-east-1
DVC_S3_ENDPOINT=http://localhost:9000

# Backend FastAPI
API_HOST=0.0.0.0
API_PORT=8000
PROJECT_ROOT=.

# Frontend Streamlit
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501
API_URL=http://api:8000
API_BASE_URL=http://api:8000
```

### 4. Configuración y Operaciones con DVC y MinIO

```bash
# 1. Inicializar DVC (si se clona por primera vez)
dvc init

# 2. Configurar remote S3 apuntando a MinIO
dvc remote add -d minio s3://dvc-storage
dvc remote modify minio endpointurl http://localhost:9000

# 3. Comprobar estado del remote
dvc remote list
dvc status

# 4. Descargar datasets y modelos desde MinIO
dvc pull

# 5. Subir datasets y modelos modificados a MinIO
dvc push
```

### 5. Despliegue con Docker Compose

El archivo `docker-compose.yml` orquesta cuatro servicios interconectados:
1. `minio`: Almacenamiento S3 de objetos persistente en `minio_data`.
2. `minio-init`: Inicialización automática e idempotente del bucket `dvc-storage`.
3. `api`: Inferencia con FastAPI escuchando en `0.0.0.0:8000`.
4. `dashboard`: Interfaz gráfica Streamlit escuchando en `0.0.0.0:8501`.

#### Comandos de Docker Compose

```bash
# Construir las imágenes y levantar todos los microservicios en segundo plano
docker compose up -d --build

# Verificar el estado y salud de los contenedores
docker compose ps

# Visualizar logs en tiempo real del backend API
docker compose logs -f api

# Visualizar logs en tiempo real del frontend Streamlit
docker compose logs -f dashboard

# Detener los servicios
docker compose down

# Detener y remover volúmenes si se desea reiniciar datos
docker compose down -v
```

### 6. Accesos y URLs de los Servicios

| Servicio | URL Local | Descripción |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Dashboard interactivo de diagnóstico predictivo y monitoreo |
| **FastAPI Root Info** | [http://localhost:8000](http://localhost:8000) | Metadatos del microservicio de inferencia |
| **FastAPI Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Estado operativo y modelo activo |
| **FastAPI Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Documentación interactiva OpenAPI / Swagger |
| **FastAPI Redoc** | [http://localhost:8000/redoc](http://localhost:8000/redoc) | Documentación estructurada Redoc |
| **MinIO S3 API** | [http://localhost:9000](http://localhost:9000) | Endpoint S3 compatible para DVC y almacenamiento |
| **MinIO Web Console** | [http://localhost:9001](http://localhost:9001) | Consola gráfica de administración de buckets y objetos |

---

## Tecnologías Utilizadas

- **Arquitectura MLOps:** Microservicios desacoplados (Docker, Docker Compose)
- **Almacenamiento y Versionamiento:** Git, DVC (Data Version Control), MinIO (S3 Compatible)
- **Backend API REST:** FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard:** Streamlit, Plotly
- **Machine Learning & Pipeline:** Scikit-Learn (Random Forest, Logistic Regression), XGBoost, Joblib
- **Lenguaje y Procesamiento:** Python 3.12, Pandas, NumPy
- **Pruebas y Calidad:** Pytest, Flake8, TestClient

---

## Autor

- **Erick Quiroz** - [GitHub Profile](https://github.com/Erick-Quiroz)
- **Diplomado en Ciencia de Datos Aplicada**