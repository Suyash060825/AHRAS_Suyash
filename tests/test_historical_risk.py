from __future__ import annotations
import unittest
from historical_risk.engine import HistoricalRiskEngine


class TestHistoricalRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HistoricalRiskEngine()

    def test_01_no_history_zero_boost(self):
        boost = self.engine.compute_history_boost("10.0.0.1")
        self.assertEqual(boost, 0.0)

    def test_02_recidivism_boost_accumulation(self):
        ip = "198.51.100.99"
        # 3 incidents, 2 alerts
        for _ in range(3):
            self.engine.record_event(ip, risk_score=0.8, is_alert=True, is_incident=True)
        for _ in range(2):
            self.engine.record_event(ip, risk_score=0.6, is_alert=True, is_incident=False)
            
        hist = self.engine.get_indicator_history(ip)
        self.assertIsNotNone(hist)
        self.assertEqual(hist.incident_count, 3)
        self.assertEqual(hist.alert_count, 5)
        
        boost = self.engine.compute_history_boost(ip, normalized_unit_scale=False)
        # incident_boost = min(30, 3*2) = 6
        # alert_boost = min(15, 5) = 5
        # recency = 1.0 -> boost = 11.0
        self.assertEqual(boost, 11.0)

    def test_03_history_boost_cap(self):
        ip = "203.0.113.50"
        for _ in range(50):
            self.engine.record_event(ip, risk_score=0.9, is_alert=True, is_incident=True)
        boost = self.engine.compute_history_boost(ip, normalized_unit_scale=False)
        self.assertEqual(boost, 45.0)  # max cap


if __name__ == "__main__":
    unittest.main()
