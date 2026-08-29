from __future__ import annotations
import unittest
import os
from evaluation.dataset_loader import DatasetLoader
from evaluation.generate_synthetic_dataset import generate_and_save
from evaluation.metrics import MetricsCalculator
from evaluation.runner import EvaluationRunner


class TestEvaluationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = generate_and_save(n_total=200)

    def test_01_dataset_loader(self):
        loader = DatasetLoader(self.csv_path)
        records = list(loader.iter_records(limit=50))
        self.assertEqual(len(records), 50)
        self.assertIn("Destination Port", records[0].features)

    def test_02_metrics_calculator(self):
        calc = MetricsCalculator()
        y_true = [1, 1, 0, 0, 1]
        y_score = [0.9, 0.8, 0.1, 0.2, 0.7]
        rep = calc.compute(y_true, y_score, threshold=0.5)
        self.assertEqual(rep.accuracy, 1.0)
        self.assertEqual(rep.f1, 1.0)
        self.assertIsNotNone(rep.brier_score)

    def test_03_deterministic_score_normalization_contract(self):
        calc = MetricsCalculator()
        
        # Test Case 1: Unit scale [0, 1] scores are preserved identically
        y_true = [0, 1, 1]
        s_unit = [0.2, 0.4, 0.8]
        rep_unit = calc.compute(y_true, s_unit, threshold=0.5)
        
        # Test Case 2: Percentage scale [0, 100] scores are deterministically converted to [0, 1]
        s_pct = [20.0, 40.0, 80.0]
        rep_pct = calc.compute(y_true, s_pct, threshold=50.0)
        
        self.assertEqual(rep_unit.precision, rep_pct.precision)
        self.assertEqual(rep_unit.recall, rep_pct.recall)
        self.assertEqual(rep_unit.f1, rep_pct.f1)
        self.assertAlmostEqual(rep_unit.brier_score, rep_pct.brier_score, places=4)

        # Test Case 3: Batches with max < 100 vs max == 100 do not alter individual score mapping
        s_batch1 = [20.0, 50.0, 90.0]
        s_batch2 = [20.0, 50.0, 100.0]
        rep_b1 = calc.compute(y_true, s_batch1, threshold=50.0)
        rep_b2 = calc.compute(y_true, s_batch2, threshold=50.0)
        self.assertEqual(rep_b1.precision, rep_b2.precision)

    def test_04_runner_execution(self):
        loader = DatasetLoader(self.csv_path)
        runner = EvaluationRunner()
        rep = runner.run_evaluation(loader, limit=30)
        self.assertGreater(rep.total_samples, 0)


if __name__ == "__main__":
    unittest.main()
