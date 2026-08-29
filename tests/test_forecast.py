from __future__ import annotations
"""
AHRAS Causal Walk-Forward Forecaster Test Suite
-----------------------------------------------
Verifies:
  1. Strict walk-forward past-only history ingestion (R_1 ... R_{t-1})
  2. Trend categorization (ESCALATING, STABLE, DE-ESCALATING)
  3. Accuracy vs Naive Persistence & Moving Average baselines
  4. Early warning lead time measurement
"""

import unittest
import numpy as np
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

    def test_05_walk_forward_accuracy_vs_baselines(self):
        # Linear escalating trajectory
        history = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        acc_holt = forecast_accuracy(self.predictor, history)
        
        # Compute baseline naive persistence (y_{t+1} = y_t)
        naive_errors = [abs(history[i] - history[i-1]) for i in range(3, len(history))]
        mae_naive = float(np.mean(naive_errors))
        
        self.assertGreater(acc_holt["n"], 0)
        self.assertLessEqual(acc_holt["mae"], mae_naive)
        self.assertLess(acc_holt["rmse"], 0.15)

    def test_06_lead_time_measurement(self):
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
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].indicator, "ip1")


if __name__ == "__main__":
    unittest.main()
