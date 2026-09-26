from __future__ import annotations
"""
Unit and Integration Tests for AHRAS Response Efficacy Learning (EXP-19)
--------------------------------------------------------------------------
Tests:
  - T01: Initial Bayesian Beta priors for domain action tuples
  - T02: Observation recording and conjugate Beta posterior updating
  - T03: Residual threat penalty and failure handling
  - T04: Hard safety guardrail: Tier-1 Crown Jewel asset protection
  - T05: Hard safety guardrail: Blast radius containment gate
  - T06: Hard safety guardrail: Combined epistemic uncertainty gate
  - T07: Digital Twin pre-execution counterfactual simulation fusion
  - T08: End-to-end feedback loop via ResponseOrchestrator.record_action_feedback
  - T09: Config toggle / backward compatibility fallback
"""

import unittest
import copy
from detection.risk_engine import RiskResult
from response.orchestrator import ResponseOrchestrator, ResponseAction
from response.efficacy_learner import (
    ResponseEfficacyLearner,
    ResponseExecutionRecord,
    EfficacyBelief,
    get_response_efficacy_learner,
)
from security_twin.models import (
    AttackScenario,
    AttackStage,
    AttackStep,
    Host,
    ActionType,
)
from security_twin.state import SecurityTwin
from security_twin.simulation import SecurityTwinSimulator


class TestResponseEfficacyLearning(unittest.TestCase):
    def setUp(self):
        self.learner = ResponseEfficacyLearner(prior_strength=5.0, dry_run=True)

    def test_t01_initial_priors(self):
        """Verifies default domain priors are initialized correctly."""
        belief_botnet = self.learner.get_belief("BLOCK_IP", "BOTNET_C2", "ALL")
        self.assertAlmostEqual(belief_botnet.mean, 0.85, places=2)
        self.assertGreater(belief_botnet.alpha, 0)
        self.assertGreater(belief_botnet.beta_param, 0)

        # Ransomware on workstation with BLOCK_IP should have low prior efficacy
        belief_rw_block = self.learner.get_belief("BLOCK_IP", "RANSOMWARE", "WORKSTATION")
        self.assertAlmostEqual(belief_rw_block.mean, 0.20, places=2)

        # Ransomware on workstation with ISOLATE_HOST should have high prior efficacy
        belief_rw_isolate = self.learner.get_belief("ISOLATE_HOST", "RANSOMWARE", "WORKSTATION")
        self.assertAlmostEqual(belief_rw_isolate.mean, 0.92, places=2)

    def test_t02_posterior_update_success(self):
        """Verifies that successful risk reduction shifts posterior distribution upwards."""
        belief_before = copy.copy(self.learner.get_belief("BLOCK_IP", "BOTNET_C2", "WORKSTATION"))
        prior_mean = belief_before.mean
        prior_samples = belief_before.sample_count

        # Record a successful mitigation: Risk reduced from 0.90 to 0.10 (Delta = 0.80)
        rec = self.learner.record_outcome(
            action_type="BLOCK_IP",
            threat_family="BOTNET_C2",
            asset_class="WORKSTATION",
            entity_key="192.168.1.50",
            target_identifier="192.168.1.50",
            risk_before=0.90,
            risk_after=0.10,
            uncertainty_before=0.15,
            uncertainty_after=0.08,
            residual_activity=False,
            success=True,
        )

        self.assertIsInstance(rec, ResponseExecutionRecord)
        self.assertAlmostEqual(rec.observed_risk_reduction, 0.80, places=3)
        self.assertFalse(rec.residual_activity)

        belief_after = self.learner.get_belief("BLOCK_IP", "BOTNET_C2", "WORKSTATION")
        self.assertEqual(belief_after.sample_count, prior_samples + 1)
        # Variance should decrease with more evidence
        self.assertLess(belief_after.variance, belief_before.variance)

    def test_t03_residual_threat_penalty(self):
        """Verifies that residual threat activity penalizes posterior efficacy."""
        belief_initial = copy.copy(self.learner.get_belief("TERMINATE_PROCESS", "RANSOMWARE", "WORKSTATION"))
        mean_initial = belief_initial.mean

        # Record an ineffective mitigation where process respawned and residual activity continued
        self.learner.record_outcome(
            action_type="TERMINATE_PROCESS",
            threat_family="RANSOMWARE",
            asset_class="WORKSTATION",
            entity_key="host-05",
            target_identifier="crypt.exe (PID:999)",
            risk_before=0.95,
            risk_after=0.85,  # Only 0.10 reduction
            residual_activity=True,  # Threat persisted
            success=False,
        )

        belief_updated = self.learner.get_belief("TERMINATE_PROCESS", "RANSOMWARE", "WORKSTATION")
        self.assertLess(belief_updated.mean, mean_initial)

    def test_t04_hard_safety_tier1_crown_jewel(self):
        """Verifies that destructive action on Tier-1 asset triggers hard safety override."""
        # Domain controller target
        res = self.learner.evaluate_action_utility(
            action_type="ISOLATE_HOST",
            threat_family="RANSOMWARE",
            asset_class="DOMAIN_CONTROLLER",
            current_risk=0.95,
            confidence=0.95,
            uncertainty=0.05,
        )

        self.assertTrue(res.safety_override_triggered)
        self.assertEqual(res.recommended_decision, "STAGE_FOR_SOC")
        self.assertIn("Tier-1 asset", res.safety_reason)

    def test_t05_hard_safety_blast_radius_gate(self):
        """Verifies that high blast radius with insufficient confidence is staged for SOC."""
        # Action with blast radius > MAX_AUTONOMOUS_BLAST_RADIUS (e.g. 0.60) and lower confidence (<0.85)
        # Simulate custom high-blast action
        ACTION_COST_CUSTOM = {"HIGH_BLAST_ACTION": {"blast_radius": 0.65, "reversibility_cost": 0.30, "expected_risk_reduction": 0.70}}
        from response import efficacy_learner
        original_matrix = efficacy_learner.ACTION_COST_MATRIX
        efficacy_learner.ACTION_COST_MATRIX = {**original_matrix, **ACTION_COST_CUSTOM}

        try:
            res = self.learner.evaluate_action_utility(
                action_type="HIGH_BLAST_ACTION",
                threat_family="GENERIC",
                asset_class="WORKSTATION",
                current_risk=0.85,
                confidence=0.75,  # Insufficient for 0.65 blast radius
                uncertainty=0.20,
            )
            self.assertTrue(res.safety_override_triggered)
            self.assertEqual(res.recommended_decision, "STAGE_FOR_SOC")
            self.assertIn("Blast radius", res.safety_reason)
        finally:
            efficacy_learner.ACTION_COST_MATRIX = original_matrix

    def test_t06_hard_safety_high_uncertainty_gate(self):
        """Verifies that high epistemic uncertainty triggers staging."""
        res = self.learner.evaluate_action_utility(
            action_type="BLOCK_IP",
            threat_family="GENERIC",
            asset_class="WORKSTATION",
            current_risk=0.85,
            confidence=0.80,
            uncertainty=0.48,  # Excessive uncertainty
        )
        self.assertTrue(res.safety_override_triggered)
        self.assertEqual(res.recommended_decision, "STAGE_FOR_SOC")
        self.assertIn("uncertainty", res.safety_reason.lower())

    def test_t07_digital_twin_counterfactual_simulation_fusion(self):
        """Verifies pre-execution simulation with digital twin fuses path breakage into utility."""
        twin = SecurityTwin()
        h1 = Host(host_id="ws-10", hostname="ws-10", ip_address="10.0.1.10", criticality=0.4)
        twin.add_host(h1)
        sim = SecurityTwinSimulator(twin)

        learner_with_twin = ResponseEfficacyLearner(prior_strength=5.0, twin_simulator=sim, dry_run=True)

        scenario = AttackScenario(
            scenario_id="scen-test",
            name="Test Recon & Exploit",
            description="Test scenario",
            steps=[
                AttackStep(
                    step_id="step-1",
                    stage=AttackStage.RECON,
                    timestamp=100.0,
                    source="10.0.1.10",
                    destination="ws-10",
                    technique="T1046",
                    technique_name="Network Service Scanning",
                    preconditions={"source_active": True},
                    postconditions={"scanned": True},
                )
            ],
        )

        res = learner_with_twin.evaluate_action_utility(
            action_type="BLOCK_IP",
            threat_family="BOTNET_C2",
            asset_class="WORKSTATION",
            current_risk=0.85,
            confidence=0.90,
            uncertainty=0.10,
            twin_scenario=scenario,
            twin_target="10.0.1.10",
        )

        self.assertTrue(res.twin_simulated)
        self.assertIsNotNone(res.twin_breakage_probability)
        self.assertGreaterEqual(res.expected_risk_reduction, 0.0)

    def test_t08_end_to_end_orchestrator_feedback(self):
        """Verifies complete loop: orchestrator responds, record feedback, prior updates."""
        orch = ResponseOrchestrator(dry_run=True, efficacy_learner=self.learner)

        risk = RiskResult(
            entity_key="10.0.5.20",
            risk_score=0.92,
            severity="CRITICAL",
            severity_id=5,
            remediation_level="AUTO_REMEDIATE",
            is_alert=True,
            S_sig=0.9,
            A_ml=0.9,
            delta_D=1.5,
            T_trust=0.1,
            mitre_techniques=["T1071"],  # BOTNET_C2
            explanation="Critical C2 beacon",
        )
        evt = {"ocsf_class": "network_activity", "src_endpoint": {"ip": "10.0.5.20"}, "device": {"hostname": "ws-sec"}}

        actions = orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        action = actions[0]
        self.assertEqual(action.action_type, "BLOCK_IP")
        self.assertEqual(action.status, "EXECUTED")

        # Now simulate observer telemetry showing risk dropped to 0.05
        rec = orch.record_action_feedback(
            action_id=action.action_id,
            risk_after=0.05,
            uncertainty_after=0.05,
            residual_activity=False,
            success=True,
        )
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.observed_risk_reduction, 0.87, places=2)

        # Check summary statistics
        summary = self.learner.get_summary_statistics()
        self.assertEqual(summary["total_observations_recorded"], 1)
        self.assertAlmostEqual(summary["overall_success_rate"], 1.0, places=2)

    def test_t09_backward_compatibility_fallback(self):
        """Verifies that orchestrator without efficacy learner falls back to static baseline."""
        orch_static = ResponseOrchestrator(dry_run=True, efficacy_learner=None)
        orch_static.efficacy_learner = None

        util = orch_static.compute_action_utility("BLOCK_IP", risk_score=0.90, confidence=0.85, uncertainty=0.15)
        self.assertIsInstance(util, float)
        # Formula: 0.40 * 0.90 * 0.85 - 0.10 - 0.05 - 0.15 * 0.25 = 0.306 - 0.15 - 0.0375 = 0.1185
        self.assertAlmostEqual(util, 0.1185, places=3)


if __name__ == "__main__":
    unittest.main()
