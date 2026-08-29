from __future__ import annotations
"""
AHRAS Experiment Matrix Divergence Properties Test Suite
--------------------------------------------------------
Proves that each stage in the E0-E12 matrix genuinely alters the risk decision
when the corresponding evidence mechanism is activated.
"""

import unittest
from detection.risk_engine import RiskConfig, AdaptiveRiskEngine


class TestExperimentMatrixProperties(unittest.TestCase):
    def setUp(self):
        self.engine = AdaptiveRiskEngine()

    def test_e7_differs_from_e6_when_graph_evidence_exists(self):
        cfg_e6 = RiskConfig(use_graph=False, use_trust=True, use_uncertainty=False)
        cfg_e7 = RiskConfig(use_graph=True, use_trust=True, use_uncertainty=False)

        ml_res = type("MockML", (), {"ensemble_score": 0.80, "confidence": 0.9})()
        res_e6 = self.engine.score_risk("host_1", [], ml_res, None, g_corr=0.30, override_config=cfg_e6)
        res_e7 = self.engine.score_risk("host_1", [], ml_res, None, g_corr=0.30, override_config=cfg_e7)

        self.assertGreater(res_e7.risk_score, res_e6.risk_score)
        self.assertEqual(res_e6.G_corr, 0.0)
        self.assertGreater(res_e7.G_corr, 0.0)

    def test_e8_differs_from_e7_when_uncertainty_is_present(self):
        cfg_e7 = RiskConfig(use_uncertainty=False, use_trust=True, use_graph=True)
        cfg_e8 = RiskConfig(use_uncertainty=True, use_trust=True, use_graph=True)

        # Disagreeing detectors (High signature vs Low ML anomaly)
        sig = [type("MockSig", (), {"severity": 5, "confidence": 0.95, "rule_name": "r1", "mitre_technique": "T1486"})()]
        ml_res = type("MockML", (), {"ensemble_score": 0.10, "confidence": 0.3})()

        res_e7 = self.engine.score_risk("host_1", sig, ml_res, None, override_config=cfg_e7)
        res_e8 = self.engine.score_risk("host_1", sig, ml_res, None, override_config=cfg_e8)

        self.assertNotEqual(res_e7.risk_score, res_e8.risk_score)
        self.assertGreater(res_e8.risk_uncertainty, 0.0)

    def test_e10_differs_from_e9_when_forecast_escalation_exists(self):
        cfg_e9 = RiskConfig(use_forecast=False, use_ti=True, use_graph=True)
        cfg_e10 = RiskConfig(use_forecast=True, use_ti=True, use_graph=True)

        ml_res = type("MockML", (), {"ensemble_score": 0.60, "confidence": 0.8})()
        res_e9 = self.engine.score_risk("host_1", [], ml_res, None, p_fore=0.20, override_config=cfg_e9)
        res_e10 = self.engine.score_risk("host_1", [], ml_res, None, p_fore=0.20, override_config=cfg_e10)

        self.assertGreater(res_e10.risk_score, res_e9.risk_score)
        self.assertEqual(res_e9.P_fore, 0.0)
        self.assertGreater(res_e10.P_fore, 0.0)


if __name__ == "__main__":
    unittest.main()
