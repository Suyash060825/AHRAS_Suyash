from __future__ import annotations
"""
AHRAS XAI Fidelity & Replayability Test Suite
---------------------------------------------
Verifies that replay_decision_trace reconstructs the engine risk score
with exact precision (error <= 1e-4) across all edge cases, clipping boundaries,
and detector configurations.
"""

import unittest
from detection.risk_engine import AdaptiveRiskEngine, RiskConfig, replay_decision_trace
from xai.fidelity_ledger import XAIFidelityLedger


class TestXAIFidelityReplay(unittest.TestCase):
    def setUp(self):
        self.engine = AdaptiveRiskEngine()
        self.ledger = XAIFidelityLedger()

    def test_zero_risk_replay(self):
        # Empty inputs -> 0.0 risk
        res = self.engine.score_risk("host_clean", [], None, None)
        self.assertIsNotNone(res.trace)
        replayed = replay_decision_trace(res.trace)
        self.assertAlmostEqual(res.risk_score, replayed, places=4)
        rec = self.ledger.verify_trace_replay(res.trace)
        self.assertTrue(rec.is_faithful)

    def test_max_score_clipping_boundary_replay(self):
        # Massive scores that exceed 1.0 clipping bound
        sig = [type("Sig", (), {"severity": 5, "confidence": 1.0, "rule_name": "r1", "mitre_technique": "T1486"})()]
        ml = type("ML", (), {"ensemble_score": 1.0, "confidence": 1.0})()
        stat = type("Stat", (), {"behavioral_drift": 3.0, "confidence": 1.0, "flags": [], "mitre_techniques": []})()

        res = self.engine.score_risk("host_breach", sig, ml, stat, h_boost=0.3, g_corr=0.3, ti_score=0.4, a_crit=1.5)
        self.assertEqual(res.risk_score, 1.0)
        self.assertIsNotNone(res.trace)
        
        replayed = replay_decision_trace(res.trace)
        self.assertEqual(replayed, 1.0)
        rec = self.ledger.verify_trace_replay(res.trace)
        self.assertTrue(rec.is_faithful)

    def test_conflicting_detectors_replay(self):
        # High signature, zero ML, moderate statistical
        sig = [type("Sig", (), {"severity": 4, "confidence": 0.9, "rule_name": "r2", "mitre_technique": "T1021"})()]
        stat = type("Stat", (), {"behavioral_drift": 1.2, "confidence": 0.6, "flags": [], "mitre_techniques": []})()

        res = self.engine.score_risk("host_conflict", sig, None, stat)
        self.assertIsNotNone(res.trace)
        replayed = replay_decision_trace(res.trace)
        self.assertAlmostEqual(res.risk_score, replayed, places=4)
        rec = self.ledger.verify_trace_replay(res.trace)
        self.assertTrue(rec.is_faithful)

    def test_missing_subsystems_replay(self):
        # Test with custom configuration disabling trust and forecast
        cfg = RiskConfig(use_trust=False, use_forecast=False)
        ml = type("ML", (), {"ensemble_score": 0.6, "confidence": 0.7})()

        res = self.engine.score_risk("host_partial", [], ml, None, override_config=cfg)
        self.assertIsNotNone(res.trace)
        replayed = replay_decision_trace(res.trace)
        self.assertAlmostEqual(res.risk_score, replayed, places=4)
        rec = self.ledger.verify_trace_replay(res.trace)
        self.assertTrue(rec.is_faithful)


if __name__ == "__main__":
    unittest.main()
