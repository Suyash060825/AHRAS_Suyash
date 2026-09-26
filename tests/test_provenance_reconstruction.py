from __future__ import annotations
"""
Unit and Integration Tests for AHRAS Provenance & Attack Scenario Reconstruction
---------------------------------------------------------------------------------
Validates:
  - 12 heterogeneous node types and 12 relation edge types.
  - Mandatory edge metadata contracts (timestamp, source, target, confidence, event_id).
  - Forward reachability and backward causal root-cause tracing in ProvenanceGraph.
  - Multi-stage attack scenario reconstruction from OCSF telemetry.
  - Unobserved / stealthy intermediate step detection (missing step reasoning).
  - Structure-preserving graph comparison metrics (precision, recall, completeness, depth, distance).
  - Dynamic configuration toggle (USE_PROVENANCE_RECONSTRUCTOR).
"""

import time
import pytest

from provenance.models import (
    ProvenanceNodeType, ProvenanceEdgeType, ProvenanceNode, ProvenanceEdge,
    ProvenanceAttackScenario, GraphQualityMetrics
)
from provenance.graph import ProvenanceGraph
from provenance.reconstructor import AttackScenarioReconstructor
from provenance.metrics import (
    compute_node_precision_recall,
    compute_edge_precision_recall,
    compute_path_completeness,
    compute_depth_similarity,
    compute_structural_distance,
    compute_comprehensive_graph_quality,
)
from config import settings


def test_provenance_schema_types_and_contracts():
    """Verifies that all 12 required node types and 12 required edge types exist and serialize."""
    expected_nodes = [
        "IP", "HOST", "USER", "ACCOUNT", "PROCESS", "FILE",
        "DOMAIN", "SOCKET", "ASSET", "ALERT", "TECHNIQUE", "INCIDENT"
    ]
    for nt in expected_nodes:
        assert hasattr(ProvenanceNodeType, nt)
        assert ProvenanceNodeType[nt].value == nt

    expected_edges = [
        "CONNECTS_TO", "EXECUTES", "READS", "WRITES", "CREATES",
        "AUTHENTICATES", "RESOLVES", "COMMUNICATES", "TARGETS",
        "USES_TECHNIQUE", "TRIGGERS", "PART_OF"
    ]
    for et in expected_edges:
        assert hasattr(ProvenanceEdgeType, et)
        assert ProvenanceEdgeType[et].value == et

    # Verify edge mandatory fields contract
    edge = ProvenanceEdge(
        edge_id="e-01",
        source="10.0.1.5",
        target="host-db-01",
        relation=ProvenanceEdgeType.CONNECTS_TO,
        timestamp=1700000000.0,
        confidence=0.95,
        event_id="ev-12345",
    )
    d = edge.to_dict()
    assert d["source"] == "10.0.1.5"
    assert d["target"] == "host-db-01"
    assert d["relation"] == "CONNECTS_TO"
    assert d["timestamp"] == 1700000000.0
    assert d["confidence"] == 0.95
    assert d["event_id"] == "ev-12345"


def test_provenance_graph_traversals():
    """Verifies forward reachability and backward causal root-cause tracing."""
    g = ProvenanceGraph(graph_id="test-graph")

    # Linear causal chain: Attacker IP -> Web Host -> Process -> Shell -> Database
    t0 = 1000.0
    g.add_node(ProvenanceNode(node_id="attacker-ip", node_type=ProvenanceNodeType.IP, name="203.0.113.10", risk_score=0.9))
    g.add_node(ProvenanceNode(node_id="host-web", node_type=ProvenanceNodeType.HOST, name="web-01", risk_score=0.4))
    g.add_node(ProvenanceNode(node_id="proc-nginx", node_type=ProvenanceNodeType.PROCESS, name="nginx", risk_score=0.2))
    g.add_node(ProvenanceNode(node_id="proc-bash", node_type=ProvenanceNodeType.PROCESS, name="bash", risk_score=0.8))
    g.add_node(ProvenanceNode(node_id="host-db", node_type=ProvenanceNodeType.HOST, name="db-01", risk_score=0.7))

    g.add_edge(ProvenanceEdge("e1", "attacker-ip", "host-web", ProvenanceEdgeType.CONNECTS_TO, t0, 0.9, "ev-1"))
    g.add_edge(ProvenanceEdge("e2", "host-web", "proc-nginx", ProvenanceEdgeType.PART_OF, t0 + 5, 0.95, "ev-2"))
    g.add_edge(ProvenanceEdge("e3", "proc-nginx", "proc-bash", ProvenanceEdgeType.CREATES, t0 + 10, 0.90, "ev-3"))
    g.add_edge(ProvenanceEdge("e4", "proc-bash", "host-db", ProvenanceEdgeType.TARGETS, t0 + 20, 0.85, "ev-4"))

    # 1. Forward reachability from attacker IP
    forward = g.forward_reachability("attacker-ip", max_depth=10)
    assert forward == {"attacker-ip", "host-web", "proc-nginx", "proc-bash", "host-db"}

    # 2. Backward causal path from compromised database
    back_paths = g.backward_causal_path("host-db", max_depth=10)
    assert len(back_paths) >= 1
    root_path = back_paths[0]
    # Root of path should be attacker-ip
    assert root_path[0].source == "attacker-ip"
    assert root_path[-1].target == "host-db"

    # 3. Simple paths
    paths = g.find_all_simple_paths("attacker-ip", "host-db")
    assert len(paths) == 1
    assert len(paths[0]) == 4


def test_attack_scenario_reconstruction_from_events():
    """Verifies end-to-end ingestion and scenario reconstruction from multi-modal events."""
    reconstructor = AttackScenarioReconstructor()

    t_base = 1700000000.0
    events = [
        # Step 1: Recon scan
        {
            "event_id": "ev-recon",
            "ocsf_class_id": 1001,
            "time": t_base,
            "src_ip": "198.51.100.99",
            "dst_ip": "10.0.1.10",
            "dst_port": 80,
            "enrichment": {"mitre_technique": "T1046", "mitre_name": "Port Scan"},
        },
        # Step 2: Web exploit
        {
            "event_id": "ev-exploit",
            "ocsf_class_id": 1001,
            "time": t_base + 30.0,
            "src_ip": "198.51.100.99",
            "dst_ip": "host-web-01",
            "dst_port": 80,
            "enrichment": {"mitre_technique": "T1190", "mitre_name": "Exploit Web App", "is_threat_intel_hit": True},
        },
        # Step 3: Shell execution
        {
            "event_id": "ev-shell",
            "ocsf_class_id": 1002,
            "time": t_base + 45.0,
            "hostname": "host-web-01",
            "actor": {"process": {"pid": 2048, "name": "bash", "cmd_line": "bash -i"}},
            "process": {"parent_pid": 1024},
            "enrichment": {"suspicious_lineage": True, "mitre_technique": "T1059.004"},
        },
        # Step 4: Lateral movement to DB
        {
            "event_id": "ev-lateral",
            "ocsf_class_id": 1001,
            "time": t_base + 90.0,
            "src_ip": "host-web-01",
            "dst_ip": "host-db-prod",
            "dst_port": 22,
            "enrichment": {"mitre_technique": "T1021.004", "mitre_name": "SSH Traversal"},
        },
        # Step 5: Ransomware encryption
        {
            "event_id": "ev-encrypt",
            "ocsf_class_id": 1003,
            "time": t_base + 120.0,
            "hostname": "host-db-prod",
            "file": {"path": "/var/data/customer.db.enc", "entropy": 7.95},
            "enrichment": {"mitre_technique": "T1486"},
        },
    ]

    graph = reconstructor.ingest_events(events)
    assert len(graph.nodes) >= 5
    assert len(graph.edges) >= 5

    scenario = reconstructor.reconstruct_scenario(graph, incident_id="INC-TEST-001")
    assert scenario.incident_id == "INC-TEST-001"
    assert "RECONNAISSANCE" in scenario.stages
    assert "INITIAL_ACCESS" in scenario.stages
    assert "LATERAL_MOVEMENT" in scenario.stages
    assert scenario.confidence > 0.80
    assert scenario.path_risk > 0.40
    assert len(scenario.edges) >= 5
    assert "INC-TEST-001" in scenario.summary


def test_missing_step_gap_detection():
    """Verifies that reconstructor identifies unobserved stealthy phases when gaps exist."""
    reconstructor = AttackScenarioReconstructor()

    # Create scenario with INITIAL_ACCESS directly jumping to LATERAL_MOVEMENT (missing EXECUTION)
    t_base = 1700000000.0
    events = [
        {
            "event_id": "ev-1",
            "ocsf_class_id": 1001,
            "time": t_base,
            "src_ip": "203.0.113.1",
            "dst_ip": "host-web",
            "dst_port": 80,
            "enrichment": {"mitre_technique": "T1190"},
        },
        {
            "event_id": "ev-2",
            "ocsf_class_id": 1001,
            "time": t_base + 60.0,
            "src_ip": "host-web",
            "dst_ip": "host-db",
            "dst_port": 22,
            "enrichment": {"mitre_technique": "T1021.004"},
        },
    ]

    graph = reconstructor.ingest_events(events)
    scenario = reconstructor.reconstruct_scenario(graph)

    # Missing step should be detected
    assert len(scenario.missing_steps) >= 1
    assert any("execution" in gap.lower() for gap in scenario.missing_steps)


def test_graph_quality_metrics_exact_match():
    """Verifies that an identical graph yields perfect precision, recall, and distance."""
    g_gt = ProvenanceGraph("gt")
    g_gt.add_node(ProvenanceNode("n1", ProvenanceNodeType.IP, "10.0.1.1"))
    g_gt.add_node(ProvenanceNode("n2", ProvenanceNodeType.HOST, "host-01"))
    g_gt.add_edge(ProvenanceEdge("e1", "n1", "n2", ProvenanceEdgeType.CONNECTS_TO, 100.0, 1.0, "ev-1"))

    sc_gt = ProvenanceAttackScenario("inc-1", ["RECON"], ["n1", "n2"], list(g_gt.edges.values()), 1.0, 0.5, [])

    metrics = compute_comprehensive_graph_quality(sc_gt, sc_gt, g_gt, g_gt)
    assert metrics.node_precision == 1.0
    assert metrics.node_recall == 1.0
    assert metrics.edge_precision == 1.0
    assert metrics.edge_recall == 1.0
    assert metrics.path_completeness == 1.0
    assert metrics.depth_similarity == 1.0
    assert metrics.structural_distance == 0.0
    assert metrics.overall_graph_f1 == 1.0


def test_graph_quality_metrics_partial_missingness():
    """Verifies graph comparison metrics when reconstructed graph is missing intermediate nodes."""
    g_gt = ProvenanceGraph("gt")
    g_gt.add_node(ProvenanceNode("n1", ProvenanceNodeType.IP, "10.0.1.1"))
    g_gt.add_node(ProvenanceNode("n2", ProvenanceNodeType.HOST, "host-01"))
    g_gt.add_node(ProvenanceNode("n3", ProvenanceNodeType.PROCESS, "bash"))
    g_gt.add_edge(ProvenanceEdge("e1", "n1", "n2", ProvenanceEdgeType.CONNECTS_TO, 100.0, 1.0, "ev-1"))
    g_gt.add_edge(ProvenanceEdge("e2", "n2", "n3", ProvenanceEdgeType.EXECUTES, 110.0, 1.0, "ev-2"))
    sc_gt = ProvenanceAttackScenario("inc-gt", ["INITIAL_ACCESS", "EXECUTION"], ["n1", "n2", "n3"], list(g_gt.edges.values()), 1.0, 0.8, [])

    # Predicted graph missed node n3 and edge e2
    g_pred = ProvenanceGraph("pred")
    g_pred.add_node(ProvenanceNode("n1", ProvenanceNodeType.IP, "10.0.1.1"))
    g_pred.add_node(ProvenanceNode("n2", ProvenanceNodeType.HOST, "host-01"))
    g_pred.add_edge(ProvenanceEdge("e1", "n1", "n2", ProvenanceEdgeType.CONNECTS_TO, 100.0, 1.0, "ev-1"))
    sc_pred = ProvenanceAttackScenario("inc-pred", ["INITIAL_ACCESS"], ["n1", "n2"], list(g_pred.edges.values()), 0.9, 0.5, [])

    metrics = compute_comprehensive_graph_quality(sc_pred, sc_gt, g_pred, g_gt, gt_critical_paths=[["n1", "n2", "n3"]])
    assert metrics.node_precision == 1.0
    assert metrics.node_recall == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert metrics.edge_precision == 1.0
    assert metrics.edge_recall == 0.5
    assert metrics.path_completeness == 0.5
    assert metrics.structural_distance > 0.0


def test_provenance_configuration_toggle():
    """Verifies that USE_PROVENANCE_RECONSTRUCTOR configuration exists and is enabled."""
    assert hasattr(settings, "USE_PROVENANCE_RECONSTRUCTOR")
    assert isinstance(settings.USE_PROVENANCE_RECONSTRUCTOR, bool)
    assert settings.USE_PROVENANCE_RECONSTRUCTOR is True
