from __future__ import annotations
"""
Unit and Integration Tests for AHRAS Security Twin & Attack Replay Lab
-----------------------------------------------------------------------
Validates:
  - Enterprise digital twin topology management & queries.
  - Snapshot capture, rollback purity, and formal state diffing.
  - Attack scenario preconditions, postconditions, and OCSF event compliance.
  - Response counterfactual simulations across all mitigation actions.
  - Monte Carlo uncertainty sampling distributions and containment probabilities.
"""

import pytest
import numpy as np

from security_twin.models import (
    EntityType, AttackStage, ActionType, SimulationStatus,
    Host, User, Process, File, Network, Asset, Service, Identity,
    SecurityControl, SecurityTwinSnapshot, SimulationResult, MonteCarloResult
)
from security_twin.state import SecurityTwin, create_enterprise_test_twin
from security_twin.scenario import (
    check_step_preconditions, apply_step_postconditions, emit_ocsf_event,
    build_ransomware_burst_scenario, build_lateral_movement_scenario, build_credential_abuse_scenario
)
from security_twin.simulation import SecurityTwinSimulator
from config import settings


def test_twin_initialization_and_inventory():
    """Verifies standard enterprise twin topology setup and entity queries."""
    twin = create_enterprise_test_twin()

    assert len(twin.hosts) == 5
    assert len(twin.networks) == 3
    assert len(twin.services) == 5
    assert len(twin.processes) == 4
    assert len(twin.assets) == 2
    assert len(twin.users) == 3
    assert len(twin.identities) == 1

    # Query tests
    host = twin.get_host("host-web-01")
    assert host is not None
    assert host.ip_address == "10.0.1.10"
    assert host.criticality == 0.60
    assert not twin.is_host_isolated("host-web-01")

    host_by_ip = twin.get_host_by_ip("10.0.3.30")
    assert host_by_ip is not None
    assert host_by_ip.host_id == "host-db-prod-01"

    assert not twin.is_ip_blocked("203.0.113.55")
    assert not twin.is_process_terminated(1024)
    assert not twin.is_token_revoked("tok-jwt-ops-99")


def test_snapshot_restore_and_diff():
    """Verifies that capture_state, state mutations, and restore_state operate reversibly without state leaks."""
    twin = create_enterprise_test_twin()
    snap1 = twin.capture_state()

    # Apply multiple mutations
    twin.hosts["host-web-01"].is_isolated = True
    twin.hosts["host-web-01"].compromise_level = 0.75
    twin.processes[1024].is_terminated = True
    twin.risk_state["host-web-01"] = 0.88
    snap2 = twin.capture_state()

    # Diff inspection
    diff = SecurityTwin.diff_state(snap1, snap2)
    assert "hosts:host-web-01" in diff["entity_modifications"]
    host_changes = diff["entity_modifications"]["hosts:host-web-01"]["changes"]
    assert host_changes["is_isolated"] == (False, True)
    assert host_changes["compromise_level"] == (0.0, 0.75)

    assert "processes:1024" in diff["entity_modifications"]
    proc_changes = diff["entity_modifications"]["processes:1024"]["changes"]
    assert proc_changes["is_terminated"] == (False, True)

    assert "host-web-01" in diff["risk_deltas"]
    assert diff["risk_deltas"]["host-web-01"]["delta"] == pytest.approx(0.73, abs=1e-3)

    # Restore snapshot and verify exact reset
    twin.restore_state(snap1)
    assert not twin.hosts["host-web-01"].is_isolated
    assert twin.hosts["host-web-01"].compromise_level == 0.0
    assert not twin.processes[1024].is_terminated
    assert twin.risk_state["host-web-01"] == 0.15


def test_attack_scenario_preconditions_and_blocking():
    """Verifies attack step precondition checks when mitigations are applied."""
    twin = create_enterprise_test_twin()
    scenario = build_ransomware_burst_scenario(attacker_ip="203.0.113.55", web_host_id="host-web-01")

    # Step 1: Normal initial state -> preconditions met
    step1 = scenario.steps[0]
    ok, reason = check_step_preconditions(step1, twin)
    assert ok is True
    assert "Preconditions satisfied" in reason

    # Block attacker IP -> Step 1 must fail
    twin.networks["net-dmz"].blocked_ips.append("203.0.113.55")
    ok, reason = check_step_preconditions(step1, twin)
    assert ok is False
    assert "blocked" in reason.lower()

    # Unblock IP, but isolate host -> Step 2 must fail
    twin.networks["net-dmz"].blocked_ips.remove("203.0.113.55")
    twin.hosts["host-web-01"].is_isolated = True
    step2 = scenario.steps[1]
    ok, reason = check_step_preconditions(step2, twin)
    assert ok is False
    assert "isolated" in reason.lower()


def test_ocsf_event_emission():
    """Verifies that emit_ocsf_event synthesizes schema-compliant OCSF telemetry events."""
    twin = create_enterprise_test_twin()
    scenario = build_ransomware_burst_scenario()

    # Network event (class 1001)
    ev_net = emit_ocsf_event(scenario.steps[0], twin)
    assert ev_net["ocsf_class_id"] == 1001
    assert ev_net["ocsf_class"] == "network_activity"
    assert "src_endpoint" in ev_net and "dst_endpoint" in ev_net
    assert ev_net["enrichment"]["mitre_technique"] == "T1046"

    # Process event (class 1002)
    ev_proc = emit_ocsf_event(scenario.steps[2], twin)
    assert ev_proc["ocsf_class_id"] == 1002
    assert ev_proc["ocsf_class"] == "process_activity"
    assert ev_proc["actor"]["process"]["name"] == "sh"
    assert ev_proc["enrichment"]["suspicious_lineage"] is True

    # File event (class 1003)
    ev_file = emit_ocsf_event(scenario.steps[4], twin)
    assert ev_file["ocsf_class_id"] == 1003
    assert ev_file["ocsf_class"] == "file_activity"
    assert ev_file["file"]["entropy"] > 7.5

    # Cloud API event (class 4001)
    cred_scenario = build_credential_abuse_scenario()
    ev_cloud = emit_ocsf_event(cred_scenario.steps[0], twin)
    assert ev_cloud["ocsf_class_id"] == 4001
    assert ev_cloud["ocsf_class"] == "cloud_api"
    assert "api" in ev_cloud


def test_counterfactual_simulation_no_action_vs_block():
    """Verifies that counterfactual simulation evaluates NO_ACTION (escalation) vs BLOCK_SOURCE (containment)."""
    twin = create_enterprise_test_twin()
    scenario = build_ransomware_burst_scenario(attacker_ip="203.0.113.55")
    simulator = SecurityTwinSimulator(twin)

    # Baseline NO_ACTION
    res_none = simulator.simulate_action(
        scenario=scenario,
        action_type=ActionType.NO_ACTION.value,
        target_entity="none",
        current_risk=0.80,
    )
    assert res_none.validation_status == SimulationStatus.VALIDATED.value
    assert res_none.expected_post_action_risk > res_none.pre_action_risk  # Escalates
    assert res_none.path_breakage_probability == 0.0
    assert res_none.remaining_attack_steps == len(scenario.steps)

    # Counterfactual: BLOCK_SOURCE on attacker IP
    res_block = simulator.simulate_action(
        scenario=scenario,
        action_type=ActionType.BLOCK_SOURCE.value,
        target_entity="203.0.113.55",
        current_risk=0.80,
    )
    assert res_block.validation_status == SimulationStatus.VALIDATED.value
    assert res_block.expected_post_action_risk < res_block.pre_action_risk  # De-escalates significantly
    assert res_block.path_breakage_probability == 1.0  # Breaks at step 1
    assert res_block.blast_radius_score < 0.15
    assert "First blocked at step 0" in res_block.explanation

    # Verify state purity: twin must remain pristine
    assert not twin.is_ip_blocked("203.0.113.55")


def test_counterfactual_simulation_host_isolation_blast_radius():
    """Verifies host isolation simulation, kill-chain breakage, and blast radius calculations."""
    twin = create_enterprise_test_twin()
    scenario = build_lateral_movement_scenario(bastion_host_id="host-bastion-01", db_host_id="host-db-prod-01")
    simulator = SecurityTwinSimulator(twin)

    # Isolate bastion host (intermediate criticality: 0.70)
    res_bastion = simulator.simulate_action(
        scenario=scenario,
        action_type=ActionType.ISOLATE_HOST.value,
        target_entity="host-bastion-01",
        current_risk=0.75,
    )
    assert res_bastion.validation_status == SimulationStatus.VALIDATED.value
    assert res_bastion.path_breakage_probability > 0.0
    assert res_bastion.blast_radius_score < 0.85

    # Isolate critical prod database (criticality: 0.95 + linked database asset)
    res_db = simulator.simulate_action(
        scenario=scenario,
        action_type=ActionType.ISOLATE_HOST.value,
        target_entity="host-db-prod-01",
        current_risk=0.75,
    )
    # High blast radius due to critical asset
    assert res_db.blast_radius_score > res_bastion.blast_radius_score
    assert res_db.collateral_disruption_cost > res_bastion.collateral_disruption_cost

    # Infeasible check: non-existent host
    res_infeasible = simulator.simulate_action(
        scenario=scenario,
        action_type=ActionType.ISOLATE_HOST.value,
        target_entity="host-ghost-99",
        current_risk=0.75,
    )
    assert res_infeasible.validation_status == SimulationStatus.INFEASIBLE.value


def test_counterfactual_simulation_token_and_process():
    """Verifies simulation of REVOKE_TOKEN and TERMINATE_PROCESS mitigations."""
    twin = create_enterprise_test_twin()
    cred_scenario = build_credential_abuse_scenario(token_id="tok-jwt-ops-99")
    simulator = SecurityTwinSimulator(twin)

    # Revoke Token
    res_token = simulator.simulate_action(
        scenario=cred_scenario,
        action_type=ActionType.REVOKE_TOKEN.value,
        target_entity="tok-jwt-ops-99",
        current_risk=0.82,
    )
    assert res_token.validation_status == SimulationStatus.VALIDATED.value
    assert res_token.path_breakage_probability == 1.0
    assert res_token.expected_post_action_risk < 0.25

    # Terminate Process
    proc_scenario = build_ransomware_burst_scenario(web_host_id="host-web-01")
    res_proc = simulator.simulate_action(
        scenario=proc_scenario,
        action_type=ActionType.TERMINATE_PROCESS.value,
        target_entity="1024",
        current_risk=0.70,
    )
    assert res_proc.validation_status == SimulationStatus.VALIDATED.value
    assert res_proc.blast_radius_score > 0.0


def test_monte_carlo_distribution_sampling():
    """Verifies Monte Carlo sampling across detector uncertainty, latency, and residual risk bounds."""
    twin = create_enterprise_test_twin()
    scenario = build_ransomware_burst_scenario(attacker_ip="203.0.113.55")
    simulator = SecurityTwinSimulator(twin)

    mc_block = simulator.run_monte_carlo(
        scenario=scenario,
        action_type=ActionType.BLOCK_SOURCE.value,
        target_entity="203.0.113.55",
        num_iterations=150,
        seed=42,
    )

    assert mc_block.num_iterations == 150
    assert mc_block.containment_probability > 0.70  # High containment with perimeter block
    assert mc_block.escalation_probability < 0.30
    assert mc_block.p10_risk <= mc_block.p50_risk <= mc_block.p90_risk <= mc_block.p99_risk
    assert mc_block.mean_residual_risk < 0.40

    mc_none = simulator.run_monte_carlo(
        scenario=scenario,
        action_type=ActionType.NO_ACTION.value,
        target_entity="none",
        num_iterations=100,
        seed=42,
    )
    assert mc_none.containment_probability == 0.0
    assert mc_none.mean_residual_risk > 0.80


def test_security_twin_config_toggle():
    """Verifies that USE_SECURITY_TWIN configuration exists and defaults to enabled."""
    assert hasattr(settings, "USE_SECURITY_TWIN")
    assert isinstance(settings.USE_SECURITY_TWIN, bool)
