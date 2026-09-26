"""
Tests for AHRAS Relational Graph & Multi-Hop Attack Path Reasoning (Frontier F / EXP-27)
-----------------------------------------------------------------------------------------
Validates:
  1. ProvenanceGraphBuilder stream ingestion across OCSF telemetry modalities.
  2. SubgraphExtractor causal backward ancestor search and forward blast radius.
  3. CausalGraphPruner noise reduction and forensic anchor preservation.
  4. RelationalPathReasoner Noisy-OR risk aggregation, kill-chain inference, and chokepoints.
"""

import time
import pytest
from provenance.models import (
    ProvenanceNodeType,
    ProvenanceEdgeType,
    ProvenanceNode,
    ProvenanceEdge,
)
from provenance.graph import ProvenanceGraph
from provenance.graph_builder import ProvenanceGraphBuilder
from provenance.subgraph_extractor import SubgraphExtractor
from provenance.causal_pruner import CausalGraphPruner
from provenance.path_reasoner import RelationalPathReasoner, AttackPathSummary


class TestProvenanceGraphBuilder:
    def test_ingest_multimodal_events(self):
        builder = ProvenanceGraphBuilder()

        # Ingest process event
        proc_evt = {
            "ocsf_class": "process_activity",
            "host_id": "host-web-01",
            "process_name": "cmd.exe",
            "pid": 2048,
            "parent_process_name": "explorer.exe",
            "command": "cmd.exe /c whoami",
            "risk_score": 0.65,
            "time": 1700000000.0,
        }
        e_proc = builder.ingest_event(proc_evt)
        assert len(e_proc) == 2  # EXECUTES and CREATES (parent)
        assert "host-web-01" in builder.graph.nodes
        assert "host-web-01:proc:cmd.exe:2048" in builder.graph.nodes

        # Ingest network event
        net_evt = {
            "ocsf_class": "network_activity",
            "src_ip": "198.51.100.42",
            "dst_ip": "10.0.1.10",
            "dst_port": 4444,
            "risk_score": 0.85,
            "time": 1700000010.0,
        }
        e_net = builder.ingest_event(net_evt)
        assert len(e_net) == 1
        assert "198.51.100.42" in builder.graph.nodes
        assert "10.0.1.10" in builder.graph.nodes

        # Ingest file event
        file_evt = {
            "ocsf_class": "file_activity",
            "host_id": "host-web-01",
            "process_name": "cmd.exe",
            "file_path": "C:\\Windows\\Temp\\payload.dll",
            "action": "write",
            "risk_score": 0.75,
            "time": 1700000020.0,
        }
        e_file = builder.ingest_event(file_evt)
        assert len(e_file) == 1
        assert "host-web-01:file:C:\\Windows\\Temp\\payload.dll" in builder.graph.nodes

        # Ingest alert event
        alert_evt = {
            "ocsf_class": "security_finding",
            "rule_name": "Suspicious_DLL_Drop",
            "target_entity": "host-web-01:file:C:\\Windows\\Temp\\payload.dll",
            "risk_score": 0.95,
            "time": 1700000025.0,
        }
        e_alert = builder.ingest_event(alert_evt)
        assert len(e_alert) == 1


class TestSubgraphExtractor:
    @pytest.fixture
    def sample_graph(self):
        g = ProvenanceGraph("test-lineage-graph")
        # Ingress -> Web Server -> DB Server -> Exfil
        nodes = [
            ProvenanceNode("ip-ext", ProvenanceNodeType.IP, "198.51.100.1", first_seen=100.0, last_seen=100.0, risk_score=0.9),
            ProvenanceNode("host-web", ProvenanceNodeType.HOST, "web-01", first_seen=105.0, last_seen=105.0, risk_score=0.5),
            ProvenanceNode("proc-sh", ProvenanceNodeType.PROCESS, "sh", first_seen=110.0, last_seen=110.0, risk_score=0.8),
            ProvenanceNode("host-db", ProvenanceNodeType.HOST, "db-01", first_seen=120.0, last_seen=120.0, risk_score=0.7),
            ProvenanceNode("alert-lateral", ProvenanceNodeType.ALERT, "LateralMovementAlert", first_seen=125.0, last_seen=125.0, risk_score=0.95),
            ProvenanceNode("host-backup", ProvenanceNodeType.HOST, "backup-srv", first_seen=130.0, last_seen=130.0, risk_score=0.1),
        ]
        for n in nodes:
            g.add_node(n)

        edges = [
            ProvenanceEdge("e1", "ip-ext", "host-web", ProvenanceEdgeType.CONNECTS_TO, 105.0, 0.9),
            ProvenanceEdge("e2", "host-web", "proc-sh", ProvenanceEdgeType.EXECUTES, 110.0, 0.95),
            ProvenanceEdge("e3", "proc-sh", "host-db", ProvenanceEdgeType.CONNECTS_TO, 120.0, 0.85),
            ProvenanceEdge("e4", "host-db", "alert-lateral", ProvenanceEdgeType.TRIGGERS, 125.0, 0.99),
            ProvenanceEdge("e5", "host-db", "host-backup", ProvenanceEdgeType.COMMUNICATES, 130.0, 0.2),
        ]
        for e in edges:
            g.add_edge(e)
        return g

    def test_extract_causal_ancestors(self, sample_graph):
        extractor = SubgraphExtractor(sample_graph)
        sub = extractor.extract_causal_neighborhood(seed_node_ids=["alert-lateral"], max_backward_hops=4, max_forward_hops=1)

        # Ancestors: host-db, proc-sh, host-web, ip-ext
        assert "alert-lateral" in sub.nodes
        assert "host-db" in sub.nodes
        assert "proc-sh" in sub.nodes
        assert "host-web" in sub.nodes
        assert "ip-ext" in sub.nodes
        # Forward impact from alert-lateral has no outgoing edges
        assert len(sub.edges) >= 4


class TestCausalGraphPruner:
    def test_prune_benign_fanout_and_preserve_anchors(self):
        g = ProvenanceGraph("noisy-graph")
        # Critical attack path: IP -> Proc -> Alert
        g.add_node(ProvenanceNode("ext-ip", ProvenanceNodeType.IP, "ext", risk_score=0.8))
        g.add_node(ProvenanceNode("proc-mal", ProvenanceNodeType.PROCESS, "mal", risk_score=0.75))
        g.add_node(ProvenanceNode("alert-1", ProvenanceNodeType.ALERT, "Alert", risk_score=0.95))

        g.add_edge(ProvenanceEdge("e-att-1", "ext-ip", "proc-mal", ProvenanceEdgeType.CONNECTS_TO, 100.0, 0.9))
        g.add_edge(ProvenanceEdge("e-att-2", "proc-mal", "alert-1", ProvenanceEdgeType.TRIGGERS, 105.0, 0.95))

        # Benign administrative process with 20 file reads (fanout noise)
        g.add_node(ProvenanceNode("proc-sys", ProvenanceNodeType.PROCESS, "systemd", risk_score=0.01))
        for i in range(20):
            fid = f"file-benign-{i}"
            g.add_node(ProvenanceNode(fid, ProvenanceNodeType.FILE, fid, risk_score=0.01))
            g.add_edge(ProvenanceEdge(f"e-noise-{i}", "proc-sys", fid, ProvenanceEdgeType.READS, 100.0, 0.99))

        pruner = CausalGraphPruner(min_risk_preserve=0.40, max_benign_fanout=5)
        pruned_g, metrics = pruner.prune(g)

        # Critical attack path must be completely intact
        assert "ext-ip" in pruned_g.nodes
        assert "proc-mal" in pruned_g.nodes
        assert "alert-1" in pruned_g.nodes
        assert "e-att-1" in pruned_g.edges
        assert "e-att-2" in pruned_g.edges

        # Compression ratio must reflect pruned noise
        assert metrics["compression_ratio"] > 0.30
        assert metrics["pruned_edges"] < metrics["raw_edges"]


class TestRelationalPathReasoner:
    def test_noisy_or_computation(self):
        reasoner = RelationalPathReasoner()
        # Single node risk
        assert reasoner.compute_noisy_or_risk([0.5]) == pytest.approx(0.5, 0.01)
        # Two independent risks: 1 - (1 - 0.5)(1 - 0.5) = 0.75
        assert reasoner.compute_noisy_or_risk([0.5, 0.5]) == pytest.approx(0.75, 0.01)
        # Monotonicity: adding risk strictly increases or preserves composite
        r3 = reasoner.compute_noisy_or_risk([0.5, 0.5, 0.4])
        assert r3 > 0.75

    def test_find_multi_hop_paths_and_chokepoints(self):
        g = ProvenanceGraph("apt-campaign-graph")
        # Ingress -> Pivot Host -> Target Database -> Exfil Alert
        nodes = [
            ProvenanceNode("ip-c2", ProvenanceNodeType.IP, "45.33.32.1", first_seen=1000.0, last_seen=1000.0, risk_score=0.85),
            ProvenanceNode("host-bastion", ProvenanceNodeType.HOST, "bastion", first_seen=1005.0, last_seen=1005.0, risk_score=0.60),
            ProvenanceNode("proc-psexec", ProvenanceNodeType.PROCESS, "psexec.exe", first_seen=1010.0, last_seen=1010.0, risk_score=0.90),
            ProvenanceNode("host-db", ProvenanceNodeType.HOST, "db-primary", first_seen=1020.0, last_seen=1020.0, risk_score=0.70),
            ProvenanceNode("alert-exfil", ProvenanceNodeType.ALERT, "ExfiltrationAlert", first_seen=1030.0, last_seen=1030.0, risk_score=0.98),
        ]
        for n in nodes:
            g.add_node(n)

        edges = [
            ProvenanceEdge("e1", "ip-c2", "host-bastion", ProvenanceEdgeType.CONNECTS_TO, 1005.0, 0.9),
            ProvenanceEdge("e2", "host-bastion", "proc-psexec", ProvenanceEdgeType.EXECUTES, 1010.0, 0.95),
            ProvenanceEdge("e3", "proc-psexec", "host-db", ProvenanceEdgeType.CONNECTS_TO, 1020.0, 0.85),
            ProvenanceEdge("e4", "host-db", "alert-exfil", ProvenanceEdgeType.TRIGGERS, 1030.0, 0.99),
        ]
        for e in edges:
            g.add_edge(e)

        reasoner = RelationalPathReasoner(min_path_risk=0.50)
        paths = reasoner.find_attack_paths(g)

        assert len(paths) >= 1
        top_path = paths[0]
        assert top_path.hop_count == 4
        assert top_path.noisy_or_risk > 0.90
        assert "RECON_INGRESS" in top_path.stages_detected
        assert "EXECUTION" in top_path.stages_detected
        assert "DETECTION_TRIGGER" in top_path.stages_detected

        # Identify chokepoints
        chokepoints = reasoner.identify_critical_chokepoints(paths, top_k=2)
        assert len(chokepoints) >= 1
        choke_node_ids = [cp[0] for cp in chokepoints]
        # Intermediate nodes (bastion, psexec, host-db) should appear as containment choke points
        assert any(nid in ["host-bastion", "proc-psexec", "host-db"] for nid in choke_node_ids)
