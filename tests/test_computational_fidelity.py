from __future__ import annotations
"""
AHRAS Phase 2 Automated Test Suite: Computational Fidelity & Multi-Path Sum-Check
---------------------------------------------------------------------------------
Strictly evaluates mathematical reconstructibility across all 10 scoring paths:
  Path 1:  Base Scoring
  Path 2:  Adaptive Weighting
  Path 3:  Multipliers
  Path 4:  Rule Boosts
  Path 5:  Contextual Evidence
  Path 6:  Historical Evidence
  Path 7:  Graph Evidence
  Path 8:  Asset / Trust / MITRE
  Path 9:  Clipping Boundaries
  Path 10: Complete Score Path
"""

import unittest
from xai.computational_fidelity import ComputationalFidelityEvaluator, DEFAULT_TOLERANCE


class TestComputationalFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = ComputationalFidelityEvaluator(tolerance=DEFAULT_TOLERANCE, seed=42)

    def test_01_base_scoring_path(self):
        metrics, evals = self.evaluator.evaluate_base_scoring(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Base scoring failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertEqual(metrics.failure_rate, 0.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)
        self.assertLessEqual(metrics.mae, DEFAULT_TOLERANCE)

    def test_02_adaptive_weighting_path(self):
        metrics, evals = self.evaluator.evaluate_adaptive_weighting(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Adaptive weighting failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_03_multipliers_path(self):
        metrics, evals = self.evaluator.evaluate_multipliers(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Multipliers path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_04_rule_boosts_path(self):
        metrics, evals = self.evaluator.evaluate_rule_boosts(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Rule boosts path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_05_contextual_evidence_path(self):
        metrics, evals = self.evaluator.evaluate_contextual_evidence(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Contextual evidence path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_06_historical_evidence_path(self):
        metrics, evals = self.evaluator.evaluate_historical_evidence(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Historical evidence path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_07_graph_evidence_path(self):
        metrics, evals = self.evaluator.evaluate_graph_evidence(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Graph evidence path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_08_asset_trust_mitre_path(self):
        metrics, evals = self.evaluator.evaluate_asset_trust_mitre(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Asset/trust/MITRE path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_09_clipping_boundaries_path(self):
        metrics, evals = self.evaluator.evaluate_clipping_boundaries(n=50)
        self.assertEqual(metrics.n_samples, 50)
        self.assertTrue(metrics.is_exact, f"Clipping boundaries path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_10_complete_score_path(self):
        metrics, evals = self.evaluator.evaluate_complete_score_path(n=100)
        self.assertEqual(metrics.n_samples, 100)
        self.assertTrue(metrics.is_exact, f"Complete score path failed exactness: max_err={metrics.max_error}")
        self.assertEqual(metrics.pass_rate, 1.0)
        self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_11_all_paths_master_run(self):
        report = self.evaluator.run_all_paths(n_per_path=30)
        self.assertTrue(report.all_paths_passed)
        self.assertEqual(len(report.path_results), 10)
        for path_name, metrics in report.path_results.items():
            self.assertTrue(metrics.is_exact, f"Path {path_name} did not achieve 100% exactness")
            self.assertLessEqual(metrics.max_error, DEFAULT_TOLERANCE)

    def test_12_tamper_detection_triggers_failure(self):
        """Asserts that simulated perturbation strictly causes fidelity failure."""
        metrics, evals = self.evaluator.evaluate_complete_score_path(n=10)
        corrupted_evals = []
        for ev in evals:
            # Deliberately corrupt reconstructed score
            tampered_reconstructed = ev.reconstructed_score + 0.15
            tampered_e_abs = abs(ev.engine_score - tampered_reconstructed)
            tampered_e_rel = tampered_e_abs / max(abs(ev.engine_score), self.evaluator.epsilon)
            corrupted_evals.append(
                self.evaluator._eval_single(
                    event_id=ev.event_id,
                    path_name="tampered_test",
                    engine_score=ev.engine_score,
                    reconstructed_score=tampered_reconstructed,
                )
            )
        from xai.computational_fidelity import compute_fidelity_metrics
        tampered_metrics = compute_fidelity_metrics(corrupted_evals, "tampered_test", "Corrupted check", DEFAULT_TOLERANCE)
        self.assertFalse(tampered_metrics.is_exact)
        self.assertGreater(tampered_metrics.failure_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
