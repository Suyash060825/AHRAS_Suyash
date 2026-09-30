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


def test_knowledge_graph_endpoints():
    # 1. Detection lineage
    resp = client.get("/api/knowledge-graph/enables/tech-t1059")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_detected"] is True
    assert len(data["detection_rules"]) >= 1

    # 2. Missing sensors
    resp_sens = client.get("/api/knowledge-graph/missing-sensors/tech-t1059")
    assert resp_sens.status_code == 200
    data_sens = resp_sens.json()
    assert "missing_sensors" in data_sens

    # 3. Sensor impact
    resp_imp = client.get("/api/knowledge-graph/sensor-impact/sensor-zeek")
    assert resp_imp.status_code == 200
    data_imp = resp_imp.json()
    assert "affected_detectors" in data_imp


def test_campaign_match_endpoint():
    payload = {
        "incident_id": "api-inc-01",
        "name": "Live Ransomware Test",
        "techniques": ["T1190", "T1059", "T1486"],
        "affected_entities": ["srv-1"],
    }
    resp = client.post("/api/campaign/match", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data
    assert len(data["matches"]) >= 1


def test_vulnerabilities_prioritize_endpoint():
    payload = {
        "active_path_entities": ["ast-dmz-01", "web-dmz-01"],
        "observed_techniques": ["T1190"],
    }
    resp = client.post("/api/vulnerabilities/prioritize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prioritized_vulnerabilities" in data
    assert len(data["prioritized_vulnerabilities"]) >= 1
    # Top vulnerability should be CVE-2021-44228 on the DMZ web server
    assert data["prioritized_vulnerabilities"][0]["cve_id"] == "CVE-2021-44228"


def test_auth_token_and_logout_flow():
    # 1. Login for token
    login_resp = client.post("/api/auth/token", json={"username": "admin", "password": "AdminSecurePass2026!"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Access protected endpoint with token
    users_resp = client.get("/api/auth/users", headers=headers)
    assert users_resp.status_code == 200
    assert len(users_resp.json()["users"]) >= 1

    # 3. Logout
    logout_resp = client.post("/api/auth/logout", headers=headers)
    assert logout_resp.status_code == 200
    assert logout_resp.json()["status"] == "SUCCESS"

    # 4. Re-access with revoked token must fail with 401
    revoked_resp = client.get("/api/auth/users", headers=headers)
    assert revoked_resp.status_code == 401


def test_api_v1_versioned_routes():
    # Test that v1 aliases work identically
    resp_v1_models = client.get("/api/v1/registry/models")
    assert resp_v1_models.status_code == 200
    assert "models" in resp_v1_models.json()

    resp_v1_iocs = client.get("/api/v1/threat-intel/iocs")
    assert resp_v1_iocs.status_code == 200


def test_entity_path_sanitization():
    # Valid entity key
    valid_resp = client.get("/entities/host-192.168.1.50/report")
    assert valid_resp.status_code == 200

    # Directory traversal or illegal characters rejected
    invalid_resp = client.get("/entities/..%2F..%2Fetc%2Fpasswd/report")
    assert invalid_resp.status_code in [400, 404]


def test_unified_response_approval_and_rejection():
    # Test approval
    appr_resp = client.post("/api/response/approve", json={"alert_id": "ALT-2026-001"})
    assert appr_resp.status_code == 200
    assert appr_resp.json()["status"] == "success"

    # Test rejection
    rej_resp = client.post("/api/response/reject", json={"alert_id": "ALT-2026-002", "reason": "Benign false trigger"})
    assert rej_resp.status_code == 200
    assert rej_resp.json()["status"] == "success"

