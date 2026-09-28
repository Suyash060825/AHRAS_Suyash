"""
Tests for AHRAS API Endpoints (Alert Intelligence, Registry, Sensor Acquisition)
---------------------------------------------------------------------------------
Validates:
  1. GET /api/registry/models
  2. GET /api/registry/detections
  3. POST /api/sensor-acquisition/plan
  4. POST /api/alerts/ingest
  5. GET /api/alert-intelligence/metrics
"""

import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


def test_registry_models_endpoint():
    response = client.get("/api/registry/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3
    model_names = [m["name"] for m in data["models"]]
    assert "IsolationForestAnomalyDetector" in model_names
    assert "MultimodalSecurityEncoder" in model_names


def test_registry_detections_endpoint():
    response = client.get("/api/registry/detections")
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert len(data["detections"]) >= 20


def test_sensor_acquisition_plan_endpoint():
    payload = {
        "event_id": "test-event-api-01",
        "threat_prior": 0.85,
        "uncertainty": 0.70,
        "asset_criticality": 1.5,
        "cpu_load": 0.30,
    }
    response = client.post("/api/sensor-acquisition/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["event_id"] == "test-event-api-01"
    assert "planned_modalities" in data
    assert data["total_estimated_latency_ms"] > 0
    assert data["total_cost_units"] > 0


def test_alert_intelligence_endpoints():
    # Ingest a test alert
    alert_payload = {
        "event_id": "api-evt-01",
        "entity_key": "host:192.168.1.50",
        "source_engine": "signature",
        "detector_name": "NET-001",
        "mitre_technique": "T1046",
        "confidence": 0.95,
        "raw_score": 0.88,
        "normalized_score": 0.88,
        "severity": "HIGH",
        "evidence_ids": ["ev-api-01"],
    }
    response = client.post("/api/alerts/ingest", json=alert_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["CLUSTERED", "DEDUPLICATED"]

    # Metrics endpoint
    resp_metrics = client.get("/api/alert-intelligence/metrics")
    assert resp_metrics.status_code == 200
    metrics_data = resp_metrics.json()
    assert "deduplication" in metrics_data
    assert metrics_data["deduplication"]["total_ingested"] >= 1
