"""Pruebas automatizadas para los endpoints de la API REST FastAPI (Fase 7.6)."""

import pandas as pd
from fastapi.testclient import TestClient
from services.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Valida que GET /health responda status 200 y formato ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "predictive-maintenance-api"
    assert "active_model" in data
    print("✓ test_health_endpoint PASÓ")


def test_models_endpoint():
    """Valida que GET /models devuelva la lista de modelos y el modelo activo."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "active_model" in data
    assert len(data["models"]) >= 3
    print(f"✓ test_models_endpoint PASÓ ({len(data['models'])} modelos detectados)")


def test_predict_endpoint_valid():
    """Valida que POST /predict procese las 44 variables y retorne probabilidad y riesgo válidos."""
    # Cargar una muestra real de X_train para garantizar tipos exactos
    df_sample = pd.read_csv("data/processed/X_train.csv", nrows=1)
    features_dict = {col: float(df_sample[col].iloc[0]) for col in df_sample.columns}

    payload = {
        "equipo": "EQ-195979",
        "features": features_dict,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["equipo"] == "EQ-195979"
    assert 0.0 <= data["probabilidad_falla_7_dias"] <= 1.0
    assert 0.0 <= data["probabilidad_porcentaje"] <= 100.0
    assert data["nivel_riesgo"] in ["Bajo", "Medio", "Alto", "Crítico"]
    assert "recomendacion" in data
    print(f"✓ test_predict_endpoint_valid PASÓ (Prob: {data['probabilidad_porcentaje']}%, Riesgo: {data['nivel_riesgo']})")


def test_predict_endpoint_missing_features():
    """Valida que POST /predict rechace payloads incompletos con código 422."""
    payload = {
        "equipo": "EQ-195979",
        "features": {"Temperatura_Proceso": 4.5},  # Faltan 43 variables
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    print("✓ test_predict_endpoint_missing_features PASÓ (Validación 422 rechazada correctamente)")


def test_set_active_model_endpoint():
    """Valida que POST /models/set-active cambie el modelo activo en producción."""
    payload = {"model_filename": "modelo_random_forest.joblib"}
    response = client.post("/models/set-active", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["active_model"] == "modelo_random_forest.joblib"
    print("✓ test_set_active_model_endpoint PASÓ")


def test_models_metrics_endpoint():
    """Valida que GET /models/metrics devuelva las métricas de validación y prueba."""
    response = client.get("/models/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "validacion" in data
    assert "prueba" in data
    assert len(data["validacion"]) >= 3
    assert len(data["prueba"]) >= 3
    print("✓ test_models_metrics_endpoint PASÓ (Métricas de Validación y Prueba verificadas)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EJECUTANDO PRUEBAS UNITARIAS DE LA API FASTAPI (FASE 7.6)")
    print("=" * 60)
    test_health_endpoint()
    test_models_endpoint()
    test_models_metrics_endpoint()
    test_predict_endpoint_valid()
    test_predict_endpoint_missing_features()
    test_set_active_model_endpoint()
    print("=" * 60)
    print("TODAS LAS PRUEBAS DE LA API PASARON EXITOSAMENTE.")
    print("=" * 60 + "\n")
