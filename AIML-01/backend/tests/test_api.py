"""Tests for the fraud scoring API."""

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_versioned_health_and_readiness_routes():
    health_response = client.get("/api/v1/health")
    readiness_response = client.get("/api/v1/readyz")

    assert health_response.status_code == 200
    assert readiness_response.status_code == 200
    assert readiness_response.json()["status"] == "ready"


def test_request_id_is_propagated():
    response = client.get("/health", headers={"X-Request-ID": "test-request-001"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-001"


def test_configured_api_key_protects_private_routes(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    unauthorized = client.get("/api/v1/cases")
    authorized = client.get(
        "/api/v1/cases",
        headers={"X-API-Key": "test-api-key"},
    )
    health = client.get("/api/v1/health")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert health.status_code == 200


def test_predict_valid_transaction():
    payload = {
        "transaction_id": "txn_test_001",
        "step": 100,
        "type": "TRANSFER",
        "amount": 150000.0,
        "nameOrig": "C1000001",
        "oldbalanceOrg": 200000.0,
        "newbalanceOrig": 50000.0,
        "nameDest": "C2000001",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 150000.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()

    assert result["transaction_id"] == "txn_test_001"
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0.0 <= result["risk_score"] <= 100.0
    assert isinstance(result["is_alert"], bool)
    assert result["model_name"] == "realtime_safe_xgboost"


def test_predict_invalid_amount():
    payload = {
        "transaction_id": "txn_test_002",
        "step": 100,
        "type": "TRANSFER",
        "amount": -10.0,
        "nameOrig": "C1000001",
        "oldbalanceOrg": 200000.0,
        "newbalanceOrig": 50000.0,
        "nameDest": "C2000001",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 150000.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_get_cases_and_alerts():
    response = client.get("/cases")
    assert response.status_code == 200
    data = response.json()
    assert "cases" in data
    assert "open_count" in data
    assert "total_count" in data
    assert data["limit"] == 100
    assert data["offset"] == 0


def test_cases_are_bounded_by_pagination_parameters():
    response = client.get("/api/v1/cases?limit=1&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert len(data["cases"]) <= 1
    assert data["limit"] == 1
    assert data["offset"] == 0


def test_get_graph_summary():
    response = client.get("/graph/summary")
    assert response.status_code == 200
    data = response.json()
    assert "graph_metrics" in data
    assert "top_fraud_linked_accounts" in data


def test_get_metrics_overview():
    response = client.get("/metrics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "fraud_rate" in data
    assert "trend_data" in data


def test_get_metrics_performance():
    response = client.get("/metrics/performance")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "realtime_safe_xgboost"
    assert "confusion_matrix" in data
    assert "average_precision" in data
    assert "temporal_windows" in data