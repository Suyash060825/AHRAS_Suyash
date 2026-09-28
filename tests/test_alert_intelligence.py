import time
import pytest
from alert_intelligence.models import (
    ExposureMetric,
    IncidentCluster,
    NetworkZone,
    RawAlert,
    TriageDecision,
    TriageLevel,
)
from alert_intelligence.deduplication import AdaptiveAlertDeduplicator
from alert_intelligence.clustering import AlertClusteringEngine
from alert_intelligence.prioritizer import ExposureAwarePrioritizer
from alert_intelligence.pipeline import AlertIntelligencePipeline


def test_models_and_evidence_preservation():
    alert = RawAlert(
        event_id="EVT-001",
        entity_key="host:srv-dc-01",
        source_engine="signature",
        detector_name="NET-011",
        severity="HIGH",
        raw_score=0.85,
        normalized_score=0.85,
        confidence=0.90,
        mitre_tactic="TA0001",
        mitre_technique="T1190",
        evidence_ids=["EVD-001", "EVD-002"]
    )
    assert alert.entity_key == "host:srv-dc-01"
    assert len(alert.evidence_ids) == 2

    cluster = IncidentCluster(
        title="Test Incident",
        entity_keys=[alert.entity_key],
        primary_entity=alert.entity_key,
        start_time=alert.timestamp,
        last_update_time=alert.timestamp,
        alerts=[alert],
        tactics_present=["TA0001"],
        techniques_present=["T1190"],
        evidence_ids=list(alert.evidence_ids)
    )
    assert cluster.cluster_id.startswith("INC-")
    assert "EVD-001" in cluster.evidence_ids

    # Add second alert with new evidence ID
    alert2 = RawAlert(
        event_id="EVT-002",
        entity_key="host:srv-dc-01",
        source_engine="ml_anomaly",
        detector_name="AUTOENCODER",
        severity="CRITICAL",
        raw_score=0.95,
        normalized_score=0.95,
        confidence=0.95,
        mitre_tactic="TA0002",
        mitre_technique="T1059",
        evidence_ids=["EVD-003"]
    )
    cluster.add_alert(alert2)
    assert len(cluster.alerts) == 2
    assert "EVD-003" in cluster.evidence_ids
    assert "TA0002" in cluster.tactics_present


def test_deduplication_basic_and_evidence_accumulation():
    dedup = AdaptiveAlertDeduplicator(window_seconds=10.0, geometric_multiplier=2.0)
    
    # Alert 1
    a1 = RawAlert(
        event_id="EVT-01",
        entity_key="ip:10.0.0.5",
        source_engine="signature",
        detector_name="DOS-SLOWLORIS",
        severity="HIGH",
        raw_score=0.80,
        normalized_score=0.80,
        confidence=0.85,
        mitre_technique="T1499",
        evidence_ids=["EVD-101"]
    )
    res1, is_new1 = dedup.process_alert(a1)
    assert is_new1 is True
    assert res1 is not None
    assert res1.evidence_ids == ["EVD-101"]

    # Duplicate Alert 2 (within window) with new evidence pointer
    a2 = RawAlert(
        event_id="EVT-02",
        entity_key="ip:10.0.0.5",
        source_engine="signature",
        detector_name="DOS-SLOWLORIS",
        severity="HIGH",
        raw_score=0.82,
        normalized_score=0.82,
        confidence=0.88,
        mitre_technique="T1499",
        evidence_ids=["EVD-102"]
    )
    res2, is_new2 = dedup.process_alert(a2)
    # Duplicate suppressed
    assert is_new2 is False
    assert res2 is None

    # Check that canonical alert preserved both evidence IDs
    flushed = dedup.flush()
    assert len(flushed) == 1
    canonical = flushed[0]
    assert canonical.suppressed_count == 1
    assert "EVD-101" in canonical.evidence_ids
    assert "EVD-102" in canonical.evidence_ids
    assert canonical.normalized_score == 0.82


def test_deduplication_geometric_emission():
    dedup = AdaptiveAlertDeduplicator(window_seconds=10.0, geometric_multiplier=2.0, min_suppress_before_emit=4)
    
    emissions = []
    # Ingest 15 identical alerts
    for i in range(15):
        a = RawAlert(
            event_id=f"EVT-{i}",
            entity_key="ip:10.0.0.5",
            source_engine="signature",
            detector_name="SYN-FLOOD",
            severity="MEDIUM",
            raw_score=0.70,
            normalized_score=0.70,
            confidence=0.80,
            mitre_technique="T1498",
            evidence_ids=[f"EVD-{i}"]
        )
        out, is_new = dedup.process_alert(a)
        if out is not None:
            emissions.append((i, is_new, out.suppressed_count))

    # Should emit at i=0 (is_new=True), i=3 (total_count=4 >= min_suppress=4), i=7 (total_count=8 >= 4*2)
    assert len(emissions) >= 2
    assert emissions[0][1] is True # First emission is new
    assert emissions[1][1] is False # Geometric milestone update


def test_clustering_spatio_temporal_and_progression():
    engine = AlertClusteringEngine(correlation_window_seconds=60.0)
    now = time.time()

    # Step 1: Initial Access on host-01
    a1 = RawAlert(
        event_id="EVT-1",
        entity_key="host:workstation-01",
        source_engine="signature",
        detector_name="PHISHING-LINK",
        severity="MEDIUM",
        raw_score=0.60,
        normalized_score=0.60,
        confidence=0.80,
        mitre_tactic="TA0001",
        mitre_technique="T1566",
        timestamp=now
    )
    c1 = engine.correlate(a1)
    assert engine.active_cluster_count == 1
    assert "TA0001" in c1.tactics_present
    initial_progression = c1.progression_score

    # Step 2: Execution on same host within 10s
    a2 = RawAlert(
        event_id="EVT-2",
        entity_key="host:workstation-01",
        source_engine="ml_anomaly",
        detector_name="POWERSHELL-ANOMALY",
        severity="HIGH",
        raw_score=0.85,
        normalized_score=0.85,
        confidence=0.90,
        mitre_tactic="TA0002",
        mitre_technique="T1059.001",
        timestamp=now + 10.0
    )
    c2 = engine.correlate(a2)
    assert c2.cluster_id == c1.cluster_id # Grouped into same incident!
    assert "TA0002" in c2.tactics_present
    assert c2.progression_score > initial_progression
    assert "Multi-Stage" in c2.title


def test_clustering_dynamic_merge():
    engine = AlertClusteringEngine(correlation_window_seconds=60.0)
    now = time.time()

    # Alert on Host A
    a_hostA = RawAlert(
        event_id="EVT-A",
        entity_key="host:host-A",
        source_engine="signature",
        detector_name="BRUTE-FORCE",
        severity="MEDIUM",
        raw_score=0.60,
        normalized_score=0.60,
        confidence=0.80,
        mitre_tactic="TA0006",
        timestamp=now
    )
    cA = engine.correlate(a_hostA)

    # Alert on Host B
    a_hostB = RawAlert(
        event_id="EVT-B",
        entity_key="host:host-B",
        source_engine="ml_anomaly",
        detector_name="EXPLOIT",
        severity="HIGH",
        raw_score=0.80,
        normalized_score=0.80,
        confidence=0.85,
        mitre_tactic="TA0008",
        timestamp=now + 5.0
    )
    cB = engine.correlate(a_hostB)
    assert engine.active_cluster_count == 2

    # Bridging Lateral Movement Alert connecting Host A and Host B
    a_bridge = RawAlert(
        event_id="EVT-BRIDGE",
        entity_key="host:host-A",
        source_engine="signature",
        detector_name="LATERAL-SMB",
        severity="HIGH",
        raw_score=0.90,
        normalized_score=0.90,
        confidence=0.92,
        mitre_tactic="TA0008",
        timestamp=now + 10.0,
        metadata={"related_entities": ["host:host-B"]}
    )
    c_merged = engine.correlate(a_bridge)
    
    # Should merge both clusters into 1
    assert engine.active_cluster_count == 1
    assert "host:host-A" in c_merged.entity_keys
    assert "host:host-B" in c_merged.entity_keys
    assert len(c_merged.alerts) == 3


def test_prioritizer_asset_criticality_and_exposure():
    prioritizer = ExposureAwarePrioritizer()
    
    # Register Tier 1 Crown Jewel in DMZ with CVE
    prioritizer.register_asset(ExposureMetric(
        entity_key="host:prod-dc-01",
        network_zone=NetworkZone.DMZ,
        asset_criticality_tier=1,
        is_internet_exposed=True,
        max_cve_cvss=9.8
    ))
    
    # Register Tier 3 Lab Workstation
    prioritizer.register_asset(ExposureMetric(
        entity_key="host:lab-pc-99",
        network_zone=NetworkZone.ISOLATED_SECURE,
        asset_criticality_tier=3,
        is_internet_exposed=False,
        max_cve_cvss=0.0
    ))

    # Cluster on Tier 1
    c_tier1 = IncidentCluster(
        title="Exploit on DC",
        entity_keys=["host:prod-dc-01"],
        primary_entity="host:prod-dc-01",
        start_time=100.0,
        last_update_time=110.0,
        alerts=[
            RawAlert(
                event_id="E1", entity_key="host:prod-dc-01", source_engine="sig",
                detector_name="D1", severity="HIGH", raw_score=0.85, normalized_score=0.85,
                confidence=0.9, mitre_tactic="TA0001", uncertainty=0.05
            )
        ],
        tactics_present=["TA0001"],
        progression_score=0.40
    )
    dec1 = prioritizer.prioritize(c_tier1)

    # Cluster on Tier 3 with identical detector score
    c_tier3 = IncidentCluster(
        title="Exploit on Lab PC",
        entity_keys=["host:lab-pc-99"],
        primary_entity="host:lab-pc-99",
        start_time=100.0,
        last_update_time=110.0,
        alerts=[
            RawAlert(
                event_id="E2", entity_key="host:lab-pc-99", source_engine="sig",
                detector_name="D1", severity="HIGH", raw_score=0.85, normalized_score=0.85,
                confidence=0.9, mitre_tactic="TA0001", uncertainty=0.05
            )
        ],
        tactics_present=["TA0001"],
        progression_score=0.40
    )
    dec3 = prioritizer.prioritize(c_tier3)

    assert dec1.priority_score > dec3.priority_score
    assert dec1.triage_level in (TriageLevel.CRITICAL, TriageLevel.HIGH)
    assert dec3.triage_level in (TriageLevel.MEDIUM, TriageLevel.LOW)


def test_prioritizer_uncertainty_dampening():
    prioritizer = ExposureAwarePrioritizer()
    
    # Highly confident detector
    c_conf = IncidentCluster(
        title="Confident Incident",
        entity_keys=["host:srv-01"],
        primary_entity="host:srv-01",
        start_time=100.0,
        last_update_time=100.0,
        alerts=[
            RawAlert(
                event_id="E1", entity_key="host:srv-01", source_engine="sig",
                detector_name="D1", severity="HIGH", raw_score=0.80, normalized_score=0.80,
                confidence=0.95, uncertainty=0.0
            )
        ]
    )
    dec_conf = prioritizer.prioritize(c_conf)

    # Highly uncertain detector (same score, but uncertainty = 1.0)
    c_uncert = IncidentCluster(
        title="Uncertain Incident",
        entity_keys=["host:srv-01"],
        primary_entity="host:srv-01",
        start_time=100.0,
        last_update_time=100.0,
        alerts=[
            RawAlert(
                event_id="E2", entity_key="host:srv-01", source_engine="sig",
                detector_name="D1", severity="HIGH", raw_score=0.80, normalized_score=0.80,
                confidence=0.50, uncertainty=1.0
            )
        ]
    )
    dec_uncert = prioritizer.prioritize(c_uncert)

    # Epistemic uncertainty dampens triage priority to prevent hasty false positive responses
    assert dec_conf.priority_score > dec_uncert.priority_score


def test_pipeline_end_to_end():
    pipeline = AlertIntelligencePipeline(dedup_window_seconds=30.0, correlation_window_seconds=120.0)
    
    # Register assets
    pipeline.register_asset_exposure(ExposureMetric(
        entity_key="host:db-master",
        network_zone=NetworkZone.INTERNAL_PROD,
        asset_criticality_tier=1,
        max_cve_cvss=8.5
    ))

    # Ingest 20 alerts: 15 duplicates of SYN flood + 5 multi-stage attacks on db-master
    alerts = []
    # 15 duplicates on web-proxy
    for i in range(15):
        alerts.append(RawAlert(
            event_id=f"EVT-SYN-{i}",
            entity_key="host:web-proxy",
            source_engine="signature",
            detector_name="SYN-FLOOD",
            severity="MEDIUM",
            raw_score=0.70,
            normalized_score=0.70,
            confidence=0.85,
            mitre_tactic="TA0040",
            mitre_technique="T1498",
            evidence_ids=[f"EVD-SYN-{i}"]
        ))

    # 5 progressive alerts on db-master
    tactics = [
        ("TA0001", "T1190", "EXPLOIT-APP"),
        ("TA0002", "T1059", "REMOTE-EXEC"),
        ("TA0004", "T1068", "PRIVESC-CVE"),
        ("TA0006", "T1003", "DUMP-CREDS"),
        ("TA0010", "T1048", "EXFIL-DATA")
    ]
    for i, (tac, tec, det) in enumerate(tactics):
        alerts.append(RawAlert(
            event_id=f"EVT-APT-{i}",
            entity_key="host:db-master",
            source_engine="hybrid",
            detector_name=det,
            severity="CRITICAL",
            raw_score=0.92,
            normalized_score=0.92,
            confidence=0.96,
            mitre_tactic=tac,
            mitre_technique=tec,
            evidence_ids=[f"EVD-APT-{i}"]
        ))

    results = pipeline.ingest_batch(alerts)
    metrics = pipeline.metrics

    # Deduplication verified: majority of SYN-FLOOD alerts were suppressed
    assert metrics["deduplication"]["total_ingested"] == 20
    assert metrics["deduplication"]["total_suppressed"] > 0
    assert metrics["deduplication"]["reduction_ratio"] > 0.50

    # Incident clusters verified
    incidents = pipeline.get_all_incidents()
    assert len(incidents) >= 2

    # Top incident must be the multi-stage APT on db-master
    top_incident = incidents[0]
    assert top_incident.primary_entity == "host:db-master"
    assert top_incident.triage_level in (TriageLevel.CRITICAL, TriageLevel.HIGH)
    assert top_incident.progression_score >= 0.80
    assert len(top_incident.tactics_present) == 5

    # Check evidence preservation
    assert len(top_incident.evidence_ids) == 5
    for i in range(5):
        assert f"EVD-APT-{i}" in top_incident.evidence_ids


def test_api_alert_intelligence_endpoints():
    from fastapi.testclient import TestClient
    from api.server import app

    client = TestClient(app)

    # 1. Ingest Raw Alert via API
    alert_payload = {
        "event_id": "EVT-API-001",
        "entity_key": "host:db-critical-01",
        "source_engine": "signature",
        "detector_name": "SQL-INJECTION",
        "severity": "CRITICAL",
        "raw_score": 0.95,
        "normalized_score": 0.95,
        "confidence": 0.98,
        "mitre_tactic": "TA0001",
        "mitre_technique": "T1190",
        "evidence_ids": ["EVD-API-101"]
    }
    resp = client.post("/api/alerts/ingest", json=alert_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "CLUSTERED"
    assert "cluster_id" in data
    cluster_id = data["cluster_id"]

    # 2. Get Incident Clusters
    resp = client.get("/api/incidents")
    assert resp.status_code == 200
    inc_data = resp.json()
    assert inc_data["total_incidents"] >= 1
    found = [c for c in inc_data["incidents"] if c["cluster_id"] == cluster_id]
    assert len(found) == 1

    # 3. Get Incident Detail
    resp = client.get(f"/api/incidents/{cluster_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["cluster_id"] == cluster_id
    assert detail["primary_entity"] == "host:db-critical-01"

    # 4. Get Metrics
    resp = client.get("/api/alert-intelligence/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert "deduplication" in metrics
    assert "active_incident_count" in metrics

