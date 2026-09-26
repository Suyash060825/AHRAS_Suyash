"""
Tests for AHRAS Conformal Selective Autonomy & Safe Response Control (Frontier H / EXP-29)
-------------------------------------------------------------------------------------------
Validates:
  1. SafetyInvariantChecker critical asset gating, blast radius budgets, and reversibility.
  2. ConformalResponseController finite-sample calibration, singleton selection, and ambiguity gating.
  3. CostSensitiveResponseEngine loss minimization and automated containment safety downgrades.
"""

import time
import pytest
import numpy as np

from response.safety_invariants import SafetyInvariantChecker, SafetyVerdict
from response.conformal_controller import ConformalResponseController, ConformalSafetyDecision
from response.cost_sensitive_policy import CostSensitiveResponseEngine, ResponseActionVerdict


class TestSafetyInvariantChecker:
    def test_normal_entity_safe_containment(self):
        checker = SafetyInvariantChecker()
        verdict = checker.evaluate_action_safety(
            action="ISOLATE_HOST",
            target_entity="workstation-bob",
            entity_metadata={"is_critical": False, "subnet": "10.0.1.0/24", "subnet_total_hosts": 50},
        )
        assert verdict.passed
        assert verdict.action_permitted
        assert not verdict.requires_human_approval
        assert verdict.compensating_action == "UNISOLATE_HOST"
        assert len(verdict.violations) == 0

    def test_critical_asset_blocks_automated_containment(self):
        checker = SafetyInvariantChecker()
        verdict = checker.evaluate_action_safety(
            action="ISOLATE_HOST",
            target_entity="dc-primary",
            entity_metadata={"is_critical": True, "role": "DOMAIN_CONTROLLER"},
        )
        assert not verdict.passed
        assert not verdict.action_permitted
        assert verdict.requires_human_approval
        assert any("CRITICAL infrastructure" in v for v in verdict.violations)

    def test_rate_limiting_barrier(self):
        checker = SafetyInvariantChecker(max_actions_per_minute=3)
        now = 1000.0
        # First 3 actions pass
        for i in range(3):
            v = checker.evaluate_action_safety("ISOLATE_HOST", f"host-{i}", current_time=now + i)
            assert v.action_permitted

        # 4th action within same minute must be blocked
        v4 = checker.evaluate_action_safety("ISOLATE_HOST", "host-3", current_time=now + 10.0)
        assert not v4.action_permitted
        assert v4.requires_human_approval
        assert any("rate limit exceeded" in viol for viol in v4.violations)

    def test_subnet_blast_radius_budget(self):
        checker = SafetyInvariantChecker(max_subnet_isolate_pct=0.10)  # Max 10%
        # Subnet has 10 hosts -> max 1 isolate allowed
        meta = {"subnet": "10.0.5.0/24", "subnet_total_hosts": 10}
        v1 = checker.evaluate_action_safety("ISOLATE_HOST", "10.0.5.1", entity_metadata=meta)
        assert v1.action_permitted

        # 2nd isolate would be 20% > 10% budget
        v2 = checker.evaluate_action_safety("ISOLATE_HOST", "10.0.5.2", entity_metadata=meta)
        assert not v2.action_permitted
        assert any("blast radius" in viol for viol in v2.violations)


class TestConformalResponseController:
    @pytest.fixture
    def calibrated_controller(self):
        ctrl = ConformalResponseController(alpha=0.05)
        # Synthetic calibration set
        probs = np.array([0.02, 0.05, 0.01, 0.10, 0.95, 0.98, 0.90, 0.99, 0.03, 0.92] * 20)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 20)
        ctrl.calibrate(probs, labels)
        return ctrl

    def test_singleton_attack_autonomous_containment(self, calibrated_controller):
        # Very high attack probability, low uncertainty
        decision = calibrated_controller.evaluate_instance(p_attack=0.99, epistemic_uncertainty=0.02)
        assert decision.is_singleton
        assert decision.prediction_set == [1]
        assert decision.is_autonomous_candidate
        assert decision.gating_recommendation == "AUTONOMOUS_CONTAINMENT"

    def test_singleton_benign_autonomous_pass(self, calibrated_controller):
        # Very low attack probability, low uncertainty
        decision = calibrated_controller.evaluate_instance(p_attack=0.01, epistemic_uncertainty=0.01)
        assert decision.is_singleton
        assert decision.prediction_set == [0]
        assert decision.is_autonomous_candidate
        assert decision.gating_recommendation == "AUTONOMOUS_PASS"

    def test_ambiguous_conformal_boundary_abstains(self, calibrated_controller):
        # Ambiguous probability near decision threshold
        decision = calibrated_controller.evaluate_instance(p_attack=0.55, epistemic_uncertainty=0.10)
        assert not decision.is_autonomous_candidate
        assert decision.gating_recommendation in {"ABSTAIN", "ESCALATE_ANALYST"}

    def test_high_uncertainty_escalates(self, calibrated_controller):
        # Strong probability but high epistemic uncertainty -> must not act autonomously
        decision = calibrated_controller.evaluate_instance(p_attack=0.98, epistemic_uncertainty=0.45)
        assert not decision.is_autonomous_candidate
        assert decision.gating_recommendation == "ESCALATE_ANALYST"


class TestCostSensitiveResponseEngine:
    def test_autonomous_arbitration_on_normal_host(self):
        engine = CostSensitiveResponseEngine()
        # Calibrate controller
        engine.conformal.calibrate(
            probabilities=np.array([0.01, 0.02, 0.98, 0.99] * 25),
            ground_truth=np.array([0, 0, 1, 1] * 25),
        )

        verdict = engine.arbitrate_response(
            event_id="evt-safe-1",
            target_entity="workstation-42",
            p_attack=0.99,
            epistemic_uncertainty=0.02,
        )
        assert verdict.dispatched_action == "AUTONOMOUS_CONTAINMENT"
        assert verdict.is_autonomous_executed
        assert verdict.safety_verdict.passed

    def test_downgrade_to_staged_on_domain_controller(self):
        engine = CostSensitiveResponseEngine()
        engine.conformal.calibrate(
            probabilities=np.array([0.01, 0.02, 0.98, 0.99] * 25),
            ground_truth=np.array([0, 0, 1, 1] * 25),
        )

        # High risk on Domain Controller -> Conformal recommends AUTONOMOUS_CONTAINMENT, but safety downgrades
        verdict = engine.arbitrate_response(
            event_id="evt-dc-attack",
            target_entity="dc-primary",
            p_attack=0.99,
            epistemic_uncertainty=0.02,
            entity_metadata={"is_critical": True, "role": "DOMAIN_CONTROLLER"},
        )
        assert verdict.dispatched_action == "STAGED_CONTAINMENT"
        assert not verdict.is_autonomous_executed
        assert verdict.safety_verdict.requires_human_approval
        assert "downgraded" in verdict.audit_notes.lower()
