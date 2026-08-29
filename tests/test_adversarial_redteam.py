from __future__ import annotations
"""
AHRAS Adversarial & Red-Team Test Suite
---------------------------------------
Verifies resilience against 8 adversarial security test vectors.
"""

import unittest
from evaluation.adversarial_suite import AdversarialRedTeamSuite


class TestAdversarialRedTeam(unittest.TestCase):
    def setUp(self):
        self.suite = AdversarialRedTeamSuite()

    def test_feature_evasion_robustness(self):
        res = self.suite.test_feature_evasion()
        self.assertTrue(res["passed"], f"Feature evasion failed: {res}")

    def test_detector_disagreement_gating(self):
        res = self.suite.test_detector_disagreement()
        self.assertTrue(res["passed"], f"Detector disagreement gating failed: {res}")

    def test_alert_flooding_throughput(self):
        res = self.suite.test_alert_flooding()
        self.assertTrue(res["passed"], f"Alert flooding throughput test failed: {res}")

    def test_feedback_poisoning_defense(self):
        res = self.suite.test_feedback_poisoning()
        self.assertTrue(res["passed"], f"Feedback poisoning defense failed: {res}")

    def test_graph_poisoning_resilience(self):
        res = self.suite.test_graph_poisoning()
        self.assertTrue(res["passed"], f"Graph poisoning resilience failed: {res}")

    def test_malformed_ocsf_handling(self):
        res = self.suite.test_malformed_ocsf()
        self.assertTrue(res["passed"], f"Malformed OCSF handling failed: {res}")

    def test_replay_attack_deduplication(self):
        res = self.suite.test_replay_attacks()
        self.assertTrue(res["passed"], f"Replay attack deduplication failed: {res}")

    def test_threat_intel_freshness_dampening(self):
        res = self.suite.test_threat_intel_poisoning()
        self.assertTrue(res["passed"], f"Threat intel dampening failed: {res}")


if __name__ == "__main__":
    unittest.main()
