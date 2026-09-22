# Prediccion de Fallas en Equipos Industriales - Area de Helados

Sistema de Machine Learning para estimar la probabilidad de falla en equipos criticos durante los siguientes 7 dias, a partir de variables operativas, telemetria y registros historicos.

---

## Arquitectura del Sistema

El proyecto esta construido bajo una arquitectura modular desacoplada basada en servicios (MLOps), separando las etapas de preparacion de datos, entrenamiento, evaluacion y servicio en produccion.

```
+-------------------------------------------------------------------------+
|                           PIPELINE DE DATOS                             |
|                                                                         |
|  [data/raw] ---> [data_cleaning] ---> [feature_engineering]             |
|                       |                         |                       |
|                       v                         v                       |
|               Observaciones Limpias      Matrices X/y (Train, Val, Test)|
|                                          con Purga Temporal 7D          |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                         ENTRENAMIENTO Y EVALUACION                      |
|                                                                         |
|  [modeling]       -> Regresion Logistica, Random Forest, XGBoost        |
|  [evaluation]     -> Validacion en Test, ROC-AUC, F1-Score, Curvas ROC  |
|  [models/]        -> Serializacion del modelo seleccionado (.joblib)    |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        DESPLIEGUE Y SERVICIO                            |
|                                                                         |
|  [services/api]       (FastAPI - Puerto 8502) -> Inferencia REST        |
|  [services/dashboard] (Streamlit - Puerto 8501) -> UI y Monitoreo       |
+-------------------------------------------------------------------------+
```

### Flujo Metodologico de Datos
1. **Limpieza y Auditoria (`services/data_cleaning`):** Validacion de tipos, tratamiento de atipicos y descarte de variables con fuga de informacion (*data leakage*).
2. **Ingenieria de Caracteristicas (`services/feature_engineering`):** Generacion de medias moviles 7D, desviaciones, ratios operativos y particion cronologica estricta con ventanas de purga de 7 dias para evitar *lookahead bias*.
3. **Modelado y Evaluacion (`services/modeling`, `services/evaluation`):** Comparacion de modelos candidatos frente al baseline, seleccion del modelo optimo y diagnostico de estabilidad.
4. **Servicio e Inferencia (`services/api`, `services/dashboard`):** Exposicion de endpoints para inferencia en tiempo real y panel visual para operadores.

---

## Herramientas y Tecnologias

- **Lenguaje Base:** Python 3.12+
- **Procesamiento y Analisis:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Visualizacion y Reportes:** Matplotlib, Seaborn, Plotly
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend / Dashboard:** Streamlit
- **Versionamiento de Datos y Modelos:** DVC (Data Version Control) con almacenamiento remoto S3 / MinIO
- **Contenedores e Infraestructura:** Docker, Docker Compose
- **Control de Versiones:** Git

---

## Estructura del Repositorio

```text
.
├── data/
│   ├── raw/                 # Datos fuente originales
│   ├── interim/             # Datos intermedios transformados
│   └── processed/           # Datasets particionados finales (X_train, y_train, etc.)
├── docker/
│   ├── api/                 # Dockerfile y configuracion del backend
│   └── dashboard/           # Dockerfile y configuracion del frontend
├── models/                  # Modelos serializados (.joblib)
├── notebooks/               # Cuadernos de experimentacion y analisis exploratorio
├── reports/                 # Figuras, curvas ROC y reportes tecnicos generados
├── results/                 # Tablas de evaluacion y diagnosticos
├── scripts/                 # Scripts de configuracion, evaluacion y soporte
├── services/
│   ├── api/                 # Microservicio de inferencia (FastAPI)
│   ├── dashboard/           # Interfaz web interactiva (Streamlit)
│   ├── data_cleaning/       # Modulo de limpieza y validacion
│   ├── eda/                 # Modulo de analisis exploratorio
│   ├── evaluation/          # Modulo de evaluacion y comparacion
│   ├── feature_engineering/ # Modulo de transformaciones y particion temporal
│   └── modeling/            # Modulo de definicion y entrenamiento de modelos
├── docker-compose.yml       # Orquestacion de contenedores para despliegue
├── main_preparacion.py      # Pipeline completo de preparacion de datos
├── main_modelado.py         # Pipeline de entrenamiento de modelos
├── main_evaluacion.py       # Pipeline de evaluacion y seleccion
└── requirements.txt         # Dependencias principales del proyecto
```

---

## Estructura de Despliegue

El despliegue se organiza en dos servicios independientes comunicados por red interna HTTP:

1. **Microservicio de Inferencia (API REST):**
   - Construido con FastAPI.
   - Expone endpoints para verificacion de estado, consulta de metadatos y prediccion de probabilidad de falla con validacion de esquema.
   - Puerto por defecto: `8502`.

2. **Microservicio de Visualizacion (Dashboard):**
   - Construido con Streamlit.
   - Consume la API REST para mostrar el diagnostico de riesgo, telemetria del equipo y distribucion de alertas.
   - Puerto por defecto: `8501`.

```
                    +--------------------+
                    |  Usuario / Planta  |
                    +---------+----------+
                              |
                              | HTTP (Puerto 8501)
                              v
                    +--------------------+
                    |     Dashboard      |
                    |    (Streamlit)     |
                    +---------+----------+
                              |
                              | HTTP (Puerto 8502)
                              v
                    +--------------------+
                    |    API Inferencia  |
                    |     (FastAPI)      |
                    +---------+----------+
                              |
                              | Carga en memoria
                              v
                    +--------------------+
                    |   Modelo Activo    |
                    |  (Random Forest)   |
                    +--------------------+
```

---

## Instrucciones de Ejecucion

### Opcion 1: Despliegue con Docker Compose (Recomendado)

Construir las imagenes y levantar los contenedores:

```bash
docker compose up --build -d
```

Verificar el estado de los servicios:

```bash
docker compose ps
```

Detener los servicios:

```bash
docker compose down
```

### Opcion 2: Ejecucion Local con Entorno Virtual

1. Crear y activar el entorno virtual de Python:

```bash
python -m venv .venv
source .venv/bin/activate   # En Linux/macOS
# .venv\Scripts\Activate.ps1 # En Windows PowerShell
```

2. Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Iniciar los servicios en terminales separadas:

```bash
# Terminal 1: Iniciar API REST
uvicorn services.api.main:app --host 0.0.0.0 --port 8502

# Terminal 2: Iniciar Dashboard
streamlit run services/dashboard/app.py --server.port 8501
```

---

## Puntos de Acceso

- **Interfaz de Usuario (Dashboard):** `http://localhost:8501`
- **Servicio de Inferencia (API):** `http://localhost:8502`
- **Documentacion OpenAPI (Swagger UI):** `http://localhost:8502/docs`
- **Verificacion de Salud (Health Check):** `http://localhost:8502/health`