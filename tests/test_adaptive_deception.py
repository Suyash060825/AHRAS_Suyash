"""
AHRAS Unit & Integration Tests — Adaptive Deception as an Information Sensor (Extension H)
"""
import pytest
import time

from deception.honeypot_manager import (
    DeceptionManager,
    DeceptionLure,
    DeceptionUtilityEvaluation,
    DeceptionTriggerFeedback,
    get_deception_manager,
    LURE_PROFILES,
)


@pytest.fixture
def manager():
    return DeceptionManager()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_deception_utility_formula(manager):
    """
    Verifies DeceptionValue = ExpectedInformationGain - DeploymentCost - OperationalRisk.
    """
    eval_res = manager.evaluate_deception_utility(
        entity_key="host-prod-db-01",
        risk_score=0.75,
        uncertainty=0.60,
        attack_path_importance=0.80,
        asset_criticality=0.90,
        candidate_lure_type="HONEY_TOKEN",
        context="cloud_api",
    )

    assert isinstance(eval_res, DeceptionUtilityEvaluation)
    assert eval_res.expected_information_gain > 0.0
    assert eval_res.deployment_cost == LURE_PROFILES["HONEY_TOKEN"]["deployment_cost"]
    assert eval_res.operational_risk == LURE_PROFILES["HONEY_TOKEN"]["operational_risk"]
    expected_net = round(
        eval_res.expected_information_gain - eval_res.deployment_cost - eval_res.operational_risk, 4
    )
    assert abs(eval_res.deception_value - expected_net) < 1e-6
    assert eval_res.is_recommended is True


def test_zero_utility_on_low_uncertainty_or_low_threat(manager):
    """
    When uncertainty is already resolved (e.g. U=0.01) or risk is negligible (R=0.05),
    lure deployment is rejected to avoid cost and noise.
    """
    eval_res = manager.evaluate_deception_utility(
        entity_key="workstation-05",
        risk_score=0.10,
        uncertainty=0.05,
        attack_path_importance=0.10,
        asset_criticality=0.20,
        candidate_lure_type="FAKE_PORT",
    )
    assert eval_res.is_recommended is False
    assert eval_res.deception_value < 0.20


def test_dynamic_lure_context_selection(manager):
    """
    Tests dynamic selection of the optimal lure based on observed telemetry context.
    """
    # 1. Cloud API context -> HONEY_TOKEN
    opt_cloud = manager.select_optimal_lure(
        entity_key="iam-user-dev",
        risk_score=0.70,
        uncertainty=0.60,
        attack_path_importance=0.70,
        asset_criticality=0.70,
        context="cloud_api",
    )
    assert opt_cloud is not None
    assert opt_cloud.lure_type == "HONEY_TOKEN"

    # 2. File activity context -> DECOY_FILE
    opt_file = manager.select_optimal_lure(
        entity_key="nas-share",
        risk_score=0.75,
        uncertainty=0.55,
        attack_path_importance=0.70,
        asset_criticality=0.75,
        context="file_activity",
    )
    assert opt_file is not None
    assert opt_file.lure_type == "DECOY_FILE"

    # 3. Network port scan context -> FAKE_PORT
    opt_net = manager.select_optimal_lure(
        entity_key="gateway-dmz",
        risk_score=0.80,
        uncertainty=0.65,
        attack_path_importance=0.85,
        asset_criticality=0.80,
        context="network_activity",
    )
    assert opt_net is not None
    assert opt_net.lure_type == "FAKE_PORT"


def test_deploy_optimal_lure(manager):
    lure = manager.deploy_optimal_lure(
        entity_key="bastion-host",
        risk_score=0.80,
        uncertainty=0.60,
        attack_path_importance=0.80,
        asset_criticality=0.90,
        context="cloud_api",
    )
    assert lure is not None
    assert isinstance(lure, DeceptionLure)
    assert lure.lure_type == "HONEY_TOKEN"
    assert lure.expected_ig > 0.0
    assert lure.deception_value > 0.0
    assert lure.is_triggered is False


def test_six_stage_deception_feedback_loop(manager):
    """
    Verifies Section 9.2:
      1. Create high-confidence EvidenceRecord
      2. Attach event/entity
      3. Update security graph
      4. Update incident status
      5. Collapse uncertainty and elevate risk
      6. Confirm attack-chain
    """
    lure = manager.deploy_optimal_lure(
        entity_key="ad-server",
        risk_score=0.75,
        uncertainty=0.50,
        attack_path_importance=0.80,
        asset_criticality=0.90,
        context="process_activity",
    )
    assert lure is not None

    feedback = manager.handle_lure_trigger(
        accessed_key=lure.lure_key,
        attacker_ip="198.51.100.99",
    )

    assert feedback is not None
    assert isinstance(feedback, DeceptionTriggerFeedback)
    assert feedback.lure_id == lure.lure_id
    assert feedback.attacker_entity == "198.51.100.99"
    # Stage 5: Uncertainty collapses, Risk elevated
    assert feedback.pre_trigger_uncertainty == 0.50
    assert feedback.post_trigger_uncertainty == 0.01
    assert feedback.uncertainty_reduction >= 0.45
    assert feedback.post_trigger_risk >= 0.95
    # Stage 3 & 4: Graph and incident updates
    assert feedback.graph_update["adversary_confirmed"] is True
    assert feedback.incident_update["severity"] == "CRITICAL"
    # Stage 6: Attack chain confirmed
    assert feedback.attack_chain_confirmed is True


def test_backward_compatibility(manager):
    """Verifies existing API methods function without regression."""
    lure = manager.deploy_lure_for_entity("test-entity", risk_score=0.75, lure_type="HONEY_TOKEN")
    assert lure is not None
    assert len(manager.get_active_lures()) == 1

    # Check interaction
    hit = manager.check_interaction(lure.lure_key, attacker_ip="10.0.0.1")
    assert hit is not None
    assert hit.is_triggered is True
    assert len(manager.get_triggered_lures()) == 1


def test_concurrent_deception_access(manager):
    import concurrent.futures

    entities = [f"entity-{i}" for i in range(20)]

    def _deploy(e):
        return manager.deploy_optimal_lure(
            entity_key=e,
            risk_score=0.80,
            uncertainty=0.60,
            attack_path_importance=0.75,
            asset_criticality=0.70,
            context="cloud_api",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        lures = list(executor.map(_deploy, entities))

    assert len(lures) == 20
    assert all(l is not None for l in lures)
    assert len(manager.get_active_lures()) == 20
