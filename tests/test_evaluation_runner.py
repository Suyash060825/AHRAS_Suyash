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

    def test_03_runner_execution(self):
        loader = DatasetLoader(self.csv_path)
        runner = EvaluationRunner()
        rep = runner.run_evaluation(loader, limit=30)
        self.assertGreater(rep.total_samples, 0)


if __name__ == "__main__":
    unittest.main()
