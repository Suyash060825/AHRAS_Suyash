from __future__ import annotations
"""
AHRAS Response Policy Gating & Counterfactual Analysis Test Suite
-----------------------------------------------------------------
Verifies action utility scoring, counterfactual analysis, rollback, and duplicate suppression.
"""

import unittest
from detection.risk_engine import RiskResult
from response.orchestrator import ResponseOrchestrator


class TestResponsePolicyGating(unittest.TestCase):
    def setUp(self):
        self.orch = ResponseOrchestrator(dry_run=True)

    def test_action_utility_calculation(self):
        # Action with high risk reduction and high confidence
        utility_high = self.orch.compute_action_utility("BLOCK_IP", risk_score=0.90, confidence=0.95, uncertainty=0.05)
        # Action with low confidence and high uncertainty
        utility_low = self.orch.compute_action_utility("ISOLATE_HOST", risk_score=0.40, confidence=0.30, uncertainty=0.70)
        
        self.assertGreater(utility_high, utility_low)
        self.assertIsInstance(utility_high, float)

    def test_counterfactual_analysis(self):
        risk = RiskResult(
            entity_key="target_srv",
            risk_score=0.85,
            severity="HIGH",
            severity_id=4,
            remediation_level="STAGE_APPROVAL",
            is_alert=True,
            S_sig=0.8,
            A_ml=0.7,
            delta_D=1.0,
            T_trust=0.2,
            explanation="High Risk Incident",
        )
        cf = self.orch.counterfactual_analysis(risk, target_threshold=0.50)
        self.assertTrue(cf["is_above_threshold"])
        self.assertIn("counterfactual_scenarios", cf)
        self.assertIn("remove_signature", cf["counterfactual_scenarios"])

    def test_action_rollback(self):
        risk = RiskResult(
            entity_key="host_rollback",
            risk_score=0.95,
            severity="CRITICAL",
            severity_id=5,
            remediation_level="AUTO_REMEDIATE",
            is_alert=True,
            S_sig=1.0,
            A_ml=0.9,
            delta_D=2.0,
            T_trust=0.0,
            explanation="Critical",
        )
        evt = {"ocsf_class": "file_activity", "device": {"hostname": "host_rollback"}}
        actions = self.orch.evaluate_and_respond(risk, evt)
        self.assertGreater(len(actions), 0)
        act_id = actions[0].action_id

        rolled_back = self.orch.rollback_action(act_id)
        self.assertTrue(rolled_back)


if __name__ == "__main__":
    unittest.main()
