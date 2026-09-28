"""
Tests for AHRAS 12-Dimensional Strategic Pareto Scorecard (Frontier J / EXP-31)
-------------------------------------------------------------------------------
Validates:
  1. Complete 12-dimensional scorecard specification integrity and targets.
  2. Strict Pareto Dominance mathematical evaluation.
  3. Pareto Dominance over Traditional SOAR and Monolithic DL baselines.
  4. End-to-end scorecard report and LaTeX table generation.
"""

import os
import json
import pytest
from evaluation.run_strategic_pareto_scorecard import (
    StrategicDimension,
    load_verified_empirical_dimensions,
    compute_pareto_dominance,
    run_exp31_scorecard,
)


class TestStrategicParetoScorecard:
    def test_scorecard_dimensions_integrity(self):
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "results"))
        dims = load_verified_empirical_dimensions(results_dir)

        assert len(dims) == 12, "Scorecard must contain exactly 12 strategic dimensions"

        for d in dims:
            assert d.name, "Dimension name cannot be empty"
            assert d.unit, "Dimension unit cannot be empty"
            assert d.direction in {"HIGHER", "LOWER"}
            assert d.is_target_met, f"Strategic target for '{d.name}' must be met"

    def test_strict_pareto_dominance_math(self):
        # Case 1: Dominant system
        d1 = [
            StrategicDimension("Metric1", "P1", "E1", "u", "HIGHER", 10.0, 5.0, 5.0, 8.0, True),
            StrategicDimension("Metric2", "P2", "E2", "u", "LOWER", 1.0, 5.0, 5.0, 2.0, True),
        ]
        dom_soar, dom_dl = compute_pareto_dominance(d1)
        assert dom_soar is True
        assert dom_dl is True

        # Case 2: Inferior on one metric -> Not strictly dominant
        d2 = [
            StrategicDimension("Metric1", "P1", "E1", "u", "HIGHER", 10.0, 5.0, 5.0, 8.0, True),
            StrategicDimension("Metric2", "P2", "E2", "u", "LOWER", 6.0, 5.0, 5.0, 2.0, False),  # 6.0 > 5.0 is worse
        ]
        dom_soar_inferior, _ = compute_pareto_dominance(d2)
        assert dom_soar_inferior is False

    def test_ahras_dominates_both_baselines(self):
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "results"))
        dims = load_verified_empirical_dimensions(results_dir)

        dom_soar, dom_dl = compute_pareto_dominance(dims)
        assert dom_soar is True, "AHRAS must strictly Pareto-dominate Traditional SOAR"
        assert dom_dl is True, "AHRAS must strictly Pareto-dominate Monolithic Deep Learning"

    def test_run_exp31_benchmark_end_to_end(self):
        report = run_exp31_scorecard()
        assert report["summary"]["targets_met"] == "12/12"
        assert report["summary"]["strictly_dominates_traditional_soar"] is True
        assert report["summary"]["strictly_dominates_monolithic_dl"] is True

        # Verify artifacts exist on disk
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "results"))
        json_path = os.path.join(results_dir, "STRATEGIC_PARETO_SCORECARD.json")
        assert os.path.exists(json_path)

        tex_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "publication", "tables", "strategic_pareto_scorecard.tex"))
        assert os.path.exists(tex_path)
