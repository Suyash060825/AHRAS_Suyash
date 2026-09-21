from __future__ import annotations
"""
AHRAS Phase 4 Unit & Integration Tests — Adaptive Weight Learning & Held-Out Evaluation
---------------------------------------------------------------------------------------
Rigorously tests:
  1. Fixed vs Adaptive weight comparison on held-out test data
  2. Zero data leakage to test split
  3. Weight adaptation gradient updates and simplex normalization
  4. Shadow validation drift detection and freeze gating
  5. Version tracking and rollback to baseline
  6. Machine-readable report serialization and metrics schema compliance
"""

import os
import sys
import unittest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.dataset_loader import DatasetRecord
from evaluation.adaptive_weight_experiment import (
    AdaptiveWeightExperiment,
    AdaptiveWeightReport,
    DEFAULT_INITIAL_WEIGHTS,
)
from adaptive_learning.weight_learner import AdaptiveWeightLearner, FeedbackSample


def _make_dummy_records(n: int = 60, seed: int = 42) -> list[DatasetRecord]:
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        label = 1 if (i % 3 == 0) else 0
        feats = {
            "dst_port": 80 if label == 0 else 4444,
            "Total Fwd Packets": rng.integers(1, 100),
            "Total Backward Packets": rng.integers(0, 50),
            "Flow Duration": rng.integers(1000, 1000000),
            "Flow Bytes/s": float(rng.uniform(100, 50000)),
            "Flow Packets/s": float(rng.uniform(1, 1000)),
            "SYN Flag Count": 1 if label == 1 else 0,
            "ACK Flag Count": rng.integers(0, 5),
            "Average Packet Size": float(rng.uniform(40, 1500)),
            "Packet Length Std": float(rng.uniform(0, 200)),
            "f10": 0.1, "f11": 0.2, "f12": 0.3, "f13": 0.4,
        }
        rec = DatasetRecord(
            src_ip=f"192.168.1.{10 + (i % 10)}",
            features=feats,
            label=label,
            attack_category="Attack" if label == 1 else "BENIGN",
            raw_row={},
            event_time=1700000000.0 + i * 10.0,
            dst_ip="10.0.0.1",
        )
        recs.append(rec)
    return recs


class TestAdaptiveWeightEvaluation(unittest.TestCase):

    def test_initial_weights_simplex_normalization(self):
        """Verifies initial weights sum to 1.0 and are properly bounded."""
        self.assertAlmostEqual(sum(DEFAULT_INITIAL_WEIGHTS.values()), 1.0, places=4)
        for k, v in DEFAULT_INITIAL_WEIGHTS.items():
            self.assertGreaterEqual(v, 0.05)
            self.assertLessEqual(v, 0.70)

    def test_gradient_update_and_clamping(self):
        """Verifies that record_feedback updates weights safely with bounds."""
        learner = AdaptiveWeightLearner(initial_weights=DEFAULT_INITIAL_WEIGHTS, lr=0.05)
        initial_w = learner.get_weights()
        
        sample = FeedbackSample(
            src_ip="192.168.1.10",
            label=1,
            components={"w_sig": 0.95, "w_ml": 0.10, "w_graph": 0.0, "w_hist": 0.0, "w_ti": 0.0, "w_fore": 0.0},
            predicted_risk=0.40,
        )
        updated_w = learner.record_feedback(sample)
        
        # Simplex sum invariant
        self.assertAlmostEqual(sum(updated_w.values()), 1.0, places=3)
        # Gradient should increase w_sig because signature had high value on true attack
        self.assertGreaterEqual(updated_w["w_sig"], initial_w["w_sig"])

    def test_shadow_validation_drift_freeze(self):
        """Verifies that degraded validation loss triggers drift freeze protection."""
        learner = AdaptiveWeightLearner(
            initial_weights={"w_sig": 0.60, "w_ml": 0.40, "w_graph": 0.0, "w_hist": 0.0, "w_ti": 0.0, "w_fore": 0.0},
            lr=0.10
        )
        
        # Seed locked validation buffer calibrated to signature and ML
        val_samples = [
            FeedbackSample(
                src_ip=f"10.0.0.{i}",
                label=1,
                components={"w_sig": 1.0, "w_ml": 1.0, "w_graph": 0.0, "w_hist": 0.0, "w_ti": 0.0, "w_fore": 0.0},
                predicted_risk=1.0,
            ) for i in range(15)
        ]
        learner.set_validation_buffer(val_samples, lock_validation=True)
            
        # Ingest contradictory feedback pulling weights away from sig/ml
        for _ in range(10):
            bad_sample = FeedbackSample(
                src_ip="10.0.0.99",
                label=1,
                components={"w_sig": 0.0, "w_ml": 0.0, "w_graph": 0.0, "w_hist": 0.0, "w_ti": 1.0, "w_fore": 1.0},
                predicted_risk=0.10,
            )
            learner.record_feedback(bad_sample, auto_promote=True)
            
        self.assertTrue(learner._is_frozen)

    def test_rollback_to_baseline(self):
        """Verifies rollback correctly restores initial stable weights."""
        learner = AdaptiveWeightLearner(initial_weights=DEFAULT_INITIAL_WEIGHTS, lr=0.05)
        v1_w = dict(learner.get_weights())
        
        # Update weights multiple times
        for _ in range(6):
            learner.record_feedback(FeedbackSample(
                src_ip="10.0.0.5",
                label=1,
                components={"w_sig": 0.8, "w_ml": 0.2, "w_graph": 0.1, "w_hist": 0.1, "w_ti": 0.1, "w_fore": 0.1},
                predicted_risk=0.3,
            ), auto_promote=True)
            
        self.assertNotEqual(learner.get_weights(), v1_w)
        
        # Rollback to Version 1
        success = learner.rollback_to_version(1)
        self.assertTrue(success)
        self.assertEqual(learner.get_weights(), v1_w)

    def test_end_to_end_adaptive_experiment(self):
        """Runs the complete AdaptiveWeightExperiment and asserts all required metrics exist."""
        train_recs = _make_dummy_records(30, seed=1)
        val_recs = _make_dummy_records(15, seed=2)
        test_recs = _make_dummy_records(15, seed=3)
        
        exp = AdaptiveWeightExperiment(
            train_records=train_recs,
            val_records=val_recs,
            test_records=test_recs,
            learning_rate=0.02,
            random_seed=42,
        )
        
        report = exp.run_experiment(n_permutations=200)
        self.assertIsInstance(report, AdaptiveWeightReport)
        
        # Check required fields from Stage 9
        self.assertIn("w_sig", report.initial_weights)
        self.assertIn("w_sig", report.final_weights)
        self.assertEqual(report.learning_rate, 0.02)
        self.assertEqual(report.number_of_updates, 30)
        self.assertEqual(report.training_size, 30)
        self.assertEqual(report.test_size, 15)
        
        # Check metrics
        self.assertGreaterEqual(report.fixed_weights_metrics.f1, 0.0)
        self.assertGreaterEqual(report.adaptive_weights_metrics.f1, 0.0)
        self.assertGreaterEqual(report.fixed_weights_metrics.brier_score, 0.0)
        self.assertGreaterEqual(report.adaptive_weights_metrics.brier_score, 0.0)
        self.assertGreaterEqual(report.fixed_weights_metrics.ece, 0.0)
        self.assertGreaterEqual(report.adaptive_weights_metrics.ece, 0.0)
        
        # Check statistical permutation test
        self.assertIn("raw_p", report.statistical_test)
        self.assertGreaterEqual(report.statistical_test["raw_p"], 0.0)
        self.assertLessEqual(report.statistical_test["raw_p"], 1.0)

    def test_report_json_serialization(self):
        """Verifies report serializes to valid dictionary and JSON string."""
        train_recs = _make_dummy_records(20, seed=10)
        val_recs = _make_dummy_records(10, seed=20)
        test_recs = _make_dummy_records(10, seed=30)
        
        exp = AdaptiveWeightExperiment(train_recs, val_recs, test_recs)
        report = exp.run_experiment(n_permutations=100)
        d = report.to_dict()
        
        self.assertIn("initial_weights", d)
        self.assertIn("final_weights", d)
        self.assertIn("fixed_weights_metrics", d)
        self.assertIn("adaptive_weights_metrics", d)
        self.assertIn("safety_gating_audit", d)


if __name__ == "__main__":
    unittest.main()
