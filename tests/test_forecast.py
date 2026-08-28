from __future__ import annotations
import unittest
from forecast.predictor import (
    AttackPredictor, ForecastResult,
    walk_forward_errors, forecast_accuracy, threshold_crossing_lead_time,
)


class TestForecastPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = AttackPredictor(horizon=5)

    def test_01_insufficient_data(self):
        res = self.predictor.predict("10.0.0.1", [0.2, 0.3])
        self.assertEqual(res.trend_label, "INSUFFICIENT_DATA")
        self.assertEqual(len(res.forecast_next), 0)

    def test_02_escalating_series(self):
        # Escalating risk curve
        history = [0.20, 0.35, 0.50, 0.65, 0.80]
        res = self.predictor.predict("10.0.0.2", history, critical_threshold=0.85)
        self.assertEqual(res.trend_label, "ESCALATING")
        self.assertTrue(res.trend > 0)
        self.assertEqual(len(res.forecast_next), 5)
        self.assertTrue(res.will_breach_critical)
        self.assertIsNotNone(res.breach_in_events)

    def test_03_stable_series(self):
        history = [0.25, 0.24, 0.26, 0.25, 0.25]
        res = self.predictor.predict("10.0.0.3", history, critical_threshold=0.85)
        self.assertEqual(res.trend_label, "STABLE")
        self.assertFalse(res.will_breach_critical)

    def test_04_deescalating_series(self):
        history = [0.90, 0.70, 0.50, 0.30, 0.15]
        res = self.predictor.predict("10.0.0.4", history)
        self.assertEqual(res.trend_label, "DE-ESCALATING")
        self.assertTrue(res.trend < 0)

    def test_05_walk_forward_accuracy(self):
        history = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
        acc = forecast_accuracy(self.predictor, history)
        self.assertGreater(acc["n"], 0)
        self.assertLess(acc["mae"], 0.20)
        self.assertLess(acc["rmse"], 0.20)

    def test_06_lead_time_measurement(self):
        # Starts low, ramps up to cross 0.85 at index 5
        series = [0.10, 0.25, 0.40, 0.55, 0.70, 0.88]
        lead_time = threshold_crossing_lead_time(self.predictor, series, threshold=0.85)
        self.assertIsNotNone(lead_time)
        self.assertGreaterEqual(lead_time, 1)

    def test_07_fleet_prediction(self):
        fleet = {
            "ip1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "ip2": [0.5, 0.5, 0.5, 0.5, 0.5],
            "ip3": [0.8, 0.6, 0.4, 0.2, 0.1],
        }
        top = self.predictor.top_escalating(fleet, n=2)
        self.assertGreater(len(top), 0)
        self.assertEqual(top[0].indicator, "ip1")


if __name__ == "__main__":
    unittest.main()
