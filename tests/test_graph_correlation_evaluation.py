from __future__ import annotations
"""
Unit and Integration Tests for Phase 5 / RQ5: Graph Correlation & Multi-Hop Campaign Reasoning
---------------------------------------------------------------------------------------------
Verifies:
  1. Enterprise network topology generation (50 hosts across 4 operational tiers)
  2. Temporal edge decay dynamics in TemporalGNN
  3. Mathematical invariants of Noisy-OR probability aggregation
  4. Subgraph multi-hop traversal and causal chain extraction
  5. Alert volume reduction (>= 60% reduction invariant)
  6. Multi-hop lateral movement F1 gain (>= 0.88 invariant vs baseline <= 0.15)
  7. Campaign detection completeness & early warning lead time
  8. End-to-end report generation and schema serializability
"""

import pytest
import numpy as np

from evaluation.graph_correlation_experiment import (
    EnterpriseTopology,
    EnterpriseTelemetrySimulator,
    GraphCorrelationExperiment,
    TelemetryEvent,
    SimulatedCampaign,
)
from graph.tgnn import TemporalGNN, AttackPathPrediction
from detection.attack_path import AttackPathReasoner, AttackPath, AttackCampaign


def test_enterprise_topology_generation():
    """Validates 50 enterprise hosts across 4 tiers with realistic roles and criticality."""
    topo = EnterpriseTopology(seed=42)
    assert len(topo.hosts) == 50
    assert len(topo.ip_to_host) == 50

    t0 = topo.get_hosts_by_tier(0)
    t1 = topo.get_hosts_by_tier(1)
    t2 = topo.get_hosts_by_tier(2)
    t3 = topo.get_hosts_by_tier(3)

    assert len(t0) == 3   # dc-01, dc-02, vault-01
    assert len(t1) == 10  # db, app, file servers
    assert len(t2) == 32  # wkstn-01 .. wkstn-32
    assert len(t3) == 5   # dmz-web, vpn-gw

    for h in t0:
        assert h.criticality >= 0.90
    for h in t2:
        assert 0.20 <= h.criticality <= 0.60


def test_temporal_edge_decay():
    """Validates that edge weights decay exponentially over elapsed time."""
    decay_rate = 0.01
    tgnn = TemporalGNN(decay_rate=decay_rate)

    t0 = 1000.0
    tgnn.record_interaction("wkstn-01", "app-srv-01", timestamp=t0, severity=1.0)
    weight_0, last_t0 = tgnn.edges[("wkstn-01", "app-srv-01")]
    assert pytest.approx(weight_0, 1e-4) == 1.0

    # 100 seconds later: decayed_weight = 1.0 * exp(-0.01 * 100) = exp(-1.0) ~ 0.3679
    # with additional severity 1.0 => ~1.3679
    t1 = t0 + 100.0
    tgnn.record_interaction("wkstn-01", "app-srv-01", timestamp=t1, severity=1.0)
    weight_1, last_t1 = tgnn.edges[("wkstn-01", "app-srv-01")]
    expected = 1.0 * np.exp(-decay_rate * 100.0) + 1.0
    assert pytest.approx(weight_1, 1e-3) == expected


def test_noisy_or_probability_aggregation():
    """Validates mathematical properties of Noisy-OR: boundedness, monotonicity, independence."""
    reasoner = AttackPathReasoner()

    # Empty list
    assert reasoner.score_path_noisy_or([]) == 0.0

    # Single risk
    assert reasoner.score_path_noisy_or([0.35]) == 0.35

    # 2 independent risks: 1 - (1 - 0.5)(1 - 0.5) = 0.75
    assert pytest.approx(reasoner.score_path_noisy_or([0.5, 0.5]), 1e-4) == 0.75

    # Monotonicity invariant: adding a hop must never decrease path risk
    r2 = reasoner.score_path_noisy_or([0.3, 0.4])
    r3 = reasoner.score_path_noisy_or([0.3, 0.4, 0.5])
    r4 = reasoner.score_path_noisy_or([0.3, 0.4, 0.5, 0.6])
    assert r2 <= r3 <= r4

    # Boundedness in [0.0, 1.0] under extreme values
    assert 0.0 <= reasoner.score_path_noisy_or([0.99, 0.99, 0.99]) <= 1.0
    assert reasoner.score_path_noisy_or([0.0, 0.0]) == 0.0


def test_subgraph_causal_path_extraction():
    """Validates that a 3-hop lateral movement chain is detected as a multi-hop candidate."""
    topo = EnterpriseTopology(seed=42)
    reasoner = AttackPathReasoner()

    # 3-hop traversal: wkstn-01 -> wkstn-02 -> app-srv-01 -> db-prod-01
    hops = [
        ("wkstn-01", "wkstn-02", 0.40, 100.0),
        ("wkstn-02", "app-srv-01", 0.45, 300.0),
        ("app-srv-01", "db-prod-01", 0.50, 600.0),
    ]

    path = reasoner.evaluate_path(
        ["wkstn-01", "wkstn-02", "app-srv-01", "db-prod-01"],
        {"wkstn-01": 0.40, "wkstn-02": 0.45, "app-srv-01": 0.50, "db-prod-01": 0.50}
    )

    assert isinstance(path, AttackPath)
    assert path.hop_count == 3
    assert path.critical_node in ("app-srv-01", "db-prod-01")
    # Noisy-OR over [0.4, 0.45, 0.5, 0.5]: 1 - 0.6*0.55*0.5*0.5 = 1 - 0.0825 = 0.9175
    assert path.path_risk > 0.90


def test_alert_volume_reduction_invariant():
    """Verifies that Stage 12 graph reasoner reduces alert volume by >= 60% vs isolated detector."""
    topo = EnterpriseTopology(seed=42)
    sim = EnterpriseTelemetrySimulator(topo, seed=42)
    events, campaigns = sim.generate_benchmark_dataset(
        n_campaigns=10,
        n_benign_events=1200,
        sim_duration_sec=43200.0,
    )

    exp = GraphCorrelationExperiment(
        event_decision_threshold=0.45,
        path_decision_threshold=0.55,
        seed=42,
    )
    report = exp.evaluate(events, campaigns)

    reduction = report["comparative_gains"]["alert_volume_reduction_pct"]
    assert reduction >= 60.0, f"Alert reduction {reduction}% is below 60% requirement"


def test_lateral_movement_f1_gain():
    """Verifies that Stage 12 achieves Lateral Movement F1 >= 0.88, far exceeding B1."""
    topo = EnterpriseTopology(seed=42)
    sim = EnterpriseTelemetrySimulator(topo, seed=42)
    events, campaigns = sim.generate_benchmark_dataset(
        n_campaigns=15,
        n_benign_events=1500,
        sim_duration_sec=43200.0,
    )

    exp = GraphCorrelationExperiment(
        event_decision_threshold=0.45,
        path_decision_threshold=0.55,
        seed=42,
    )
    report = exp.evaluate(events, campaigns)

    b1_f1 = report["baseline_isolated_detector_b1"]["lateral_movement_f1"]
    st12_f1 = report["graph_correlated_reasoner_stage_12"]["lateral_movement_f1"]

    assert b1_f1 <= 0.15, f"Isolated baseline lateral movement F1 {b1_f1} should be low"
    assert st12_f1 >= 0.88, f"Stage 12 lateral movement F1 {st12_f1} must be >= 0.88"


def test_campaign_detection_completeness():
    """Verifies that graph correlation attributes >= 80% of compromised hosts in detected campaigns."""
    topo = EnterpriseTopology(seed=42)
    sim = EnterpriseTelemetrySimulator(topo, seed=42)
    events, campaigns = sim.generate_benchmark_dataset(
        n_campaigns=10,
        n_benign_events=1000,
        sim_duration_sec=36000.0,
    )

    exp = GraphCorrelationExperiment(path_decision_threshold=0.55, seed=42)
    report = exp.evaluate(events, campaigns)

    completeness = report["comparative_gains"]["campaign_completeness_pct"]
    assert completeness >= 80.0, f"Campaign completeness {completeness}% is below 80%"


def test_end_to_end_graph_correlation_runner():
    """Verifies end-to-end evaluation execution, statistical tests, and report structure."""
    topo = EnterpriseTopology(seed=123)
    sim = EnterpriseTelemetrySimulator(topo, seed=123)
    events, campaigns = sim.generate_benchmark_dataset(
        n_campaigns=8,
        n_benign_events=800,
        sim_duration_sec=28800.0,
    )

    exp = GraphCorrelationExperiment(path_decision_threshold=0.55, seed=123)
    report = exp.evaluate(events, campaigns)

    assert "experiment_id" in report
    assert report["experiment_id"] == "EXP-05"
    assert "baseline_isolated_detector_b1" in report
    assert "graph_correlated_reasoner_stage_12" in report
    assert "comparative_gains" in report
    assert "statistical_significance" in report

    stat = report["statistical_significance"]
    assert stat["statistically_significant"] is True
    assert stat["two_sided_p_value"] < 0.05
    assert len(stat["alert_reduction_95_ci"]) == 2
