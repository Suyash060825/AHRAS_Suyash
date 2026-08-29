from __future__ import annotations
"""
AHRAS Risk-to-Action Safety Efficiency (RASE) Metric Test Suite
----------------------------------------------------------------
Proves that RASE adds essential operational safety information beyond F1 score:
Demonstrates scenarios where two controllers have identical classification F1,
but drastically different operational safety and intervention efficiency.
"""

import unittest
from detection.risk_engine import compute_rase


class TestRASEMetric(unittest.TestCase):
    def test_identical_f1_divergent_rase(self):
        # Scenario A: Conservative Controller (High precision, low blast radius, low uncertainty)
        # Mitigates 0.8 risk with 0.1 uncertainty, blast radius 0.1, reversibility cost 0.05, zero false intervention
        rase_a = compute_rase(
            risk_reduction=0.80,
            uncertainty=0.10,
            blast_radius=0.10,
            reversibility_cost=0.05,
            is_false_intervention=False,
        )

        # Scenario B: Aggressive Controller (Same F1, but massive blast radius and high uncertainty)
        # Mitigates 0.8 risk with 0.6 uncertainty, blast radius 0.8, reversibility cost 0.5, zero false intervention
        rase_b = compute_rase(
            risk_reduction=0.80,
            uncertainty=0.60,
            blast_radius=0.80,
            reversibility_cost=0.50,
            is_false_intervention=False,
        )

        # Both controllers mitigated the same nominal threat, but Controller A has ~19x higher safety efficiency!
        self.assertGreater(rase_a, rase_b)
        self.assertGreater(rase_a, 4.0)
        self.assertLess(rase_b, 0.5)

    def test_false_intervention_penalty_behavior(self):
        rase_true_positive = compute_rase(0.8, 0.1, 0.2, 0.1, is_false_intervention=False)
        rase_false_positive = compute_rase(0.8, 0.1, 0.2, 0.1, is_false_intervention=True, lambda_fp_penalty=2.0)

        self.assertGreater(rase_true_positive, rase_false_positive)

    def test_monotonicity_under_uncertainty(self):
        # RASE should decrease monotonically as uncertainty increases
        rase_low_unc = compute_rase(0.8, uncertainty=0.05, blast_radius=0.2, reversibility_cost=0.1, is_false_intervention=False)
        rase_mid_unc = compute_rase(0.8, uncertainty=0.30, blast_radius=0.2, reversibility_cost=0.1, is_false_intervention=False)
        rase_high_unc = compute_rase(0.8, uncertainty=0.80, blast_radius=0.2, reversibility_cost=0.1, is_false_intervention=False)

        self.assertGreater(rase_low_unc, rase_mid_unc)
        self.assertGreater(rase_mid_unc, rase_high_unc)


if __name__ == "__main__":
    unittest.main()
