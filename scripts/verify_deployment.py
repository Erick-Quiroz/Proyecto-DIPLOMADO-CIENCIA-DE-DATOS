"""Script de verificación integral de los microservicios desplegados con Docker Compose."""

import json
import urllib.request
import pandas as pd


def verify_all():
    print("=" * 65)
    print("VERIFICACIÓN DE MICROSERVICIOS EN PRODUCCIÓN (DOCKER COMPOSE)")
    print("=" * 65)

    api_port = "8502"

    # 1. Healthcheck FastAPI
    with urllib.request.urlopen(f"http://localhost:{api_port}/health") as resp:
        data = json.loads(resp.read().decode())
        print(f"✓ 1. Healthcheck FastAPI (/health en :{api_port}): Status={data['status']}, Modelo={data['active_model']}")
        assert data["status"] == "ok"
        assert data["active_model"] == "modelo_random_forest.joblib"

    # 2. Catálogo de Modelos
    with urllib.request.urlopen(f"http://localhost:{api_port}/models") as resp:
        data = json.loads(resp.read().decode())
        print(f"✓ 2. Catálogo (/models): {len(data['models'])} modelos disponibles en runtime")
        assert len(data["models"]) >= 3

    # 3. Inferencia con Datos Reales
    df_sample = pd.read_csv("data/processed/X_train.csv", nrows=1)
    features_dict = {col: float(df_sample[col].iloc[0]) for col in df_sample.columns}
    payload = json.dumps({"equipo": "EQ-195979", "features": features_dict}).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{api_port}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"✓ 3. Inferencia (/predict):")
        print(f"     - Equipo: {data['equipo']}")
        print(f"     - Probabilidad de falla (7 días): {data['probabilidad_porcentaje']}%")
        print(f"     - Nivel de Riesgo: {data['nivel_riesgo']}")
        print(f"     - Acción Recomendada: {data['recomendacion']}")
        assert 0.0 <= data["probabilidad_falla_7_dias"] <= 1.0

    # 4. Documentación OpenAPI / Swagger
    with urllib.request.urlopen(f"http://localhost:{api_port}/docs") as resp:
        print(f"✓ 4. Swagger UI (/docs en :{api_port}): HTTP {resp.status} OK")
        assert resp.status == 200

    # 5. Streamlit Health
    with urllib.request.urlopen("http://localhost:8501/_stcore/health") as resp:
        print(f"✓ 5. Streamlit Dashboard (Port 8501): HTTP {resp.status} OK")
        assert resp.status == 200

    print("=" * 65)
    print("TODAS LAS PRUEBAS DE INFRAESTRUCTURA Y SERVICIOS PASARON EXITOSAMENTE.")
    print("=" * 65)


if __name__ == "__main__":
    verify_all()
