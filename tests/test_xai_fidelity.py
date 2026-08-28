from __future__ import annotations
import unittest
from xai.fidelity_ledger import XAIFidelityLedger, XAIFidelityRecord


class TestXAIFidelity(unittest.TestCase):
    def setUp(self):
        self.ledger = XAIFidelityLedger(tolerance=0.01)

    def test_01_exact_sum_check_faithful(self):
        components = [
            {"name": "signature", "contribution": 0.40},
            {"name": "anomaly", "contribution": 0.30},
        ]
        adjustments = [
            {"type": "trust_discount", "value": -0.10},
        ]
        # Sum = 0.40 + 0.30 - 0.10 = 0.60
        rec = self.ledger.verify_explanation(
            event_id="EVT-001",
            entity_key="10.0.0.5",
            engine_risk_score=0.60,
            components=components,
            adjustments=adjustments,
        )
        self.assertTrue(rec.is_faithful)
        self.assertAlmostEqual(rec.reconstruction_error, 0.0, places=5)

    def test_02_exact_sum_check_unfaithful(self):
        components = [
            {"name": "signature", "contribution": 0.10},
        ]
        # Sum = 0.10, but engine score = 0.85
        rec = self.ledger.verify_explanation(
            event_id="EVT-002",
            entity_key="10.0.0.6",
            engine_risk_score=0.85,
            components=components,
            adjustments=[],
        )
        self.assertFalse(rec.is_faithful)
        self.assertAlmostEqual(rec.reconstruction_error, 0.75, places=5)

    def test_03_domain_feature_alignment(self):
        rec = self.ledger.verify_explanation(
            event_id="EVT-003",
            entity_key="10.0.0.7",
            engine_risk_score=0.90,
            components=[{"name": "sig", "contribution": 0.90}],
            adjustments=[],
            attack_type="port_scan",
            top_explained_features=["unique_dst_ports", "packet_count", "tcp_flags"],
        )
        self.assertIsNotNone(rec.fap)
        self.assertIsNotNone(rec.far)
        self.assertGreater(rec.fap, 0.5)

    def test_04_ledger_summary(self):
        summary = self.ledger.get_summary()
        self.assertIn("total_checked", summary)
        self.assertIn("fidelity_rate", summary)


if __name__ == "__main__":
    unittest.main()
