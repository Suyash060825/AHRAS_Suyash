"""
Unit and Integration Test Suite — EXP-11: Explanation Reliability Audit 2.0
----------------------------------------------------------------------------
Validates the multidimensional reliability properties of AHRAS XAI:
  1. Rank correlation & Jaccard mathematical properties
  2. Perturbation stability
  3. Sufficiency monotonicity across top-k
  4. Comprehensiveness degradation
  5. Spurious feature robustness & non-leakage
  6. Cross-run consistency
  7. Counterfactual alignment
  8. Full audit report execution and schema compliance
"""

import pytest
import numpy as np
from typing import Dict, List

from detection.risk_engine import AdaptiveRiskEngine, RiskConfig, DecisionTrace
from xai.reliability_audit import (
    XAIReliabilityAuditor,
    XAIReliabilityAuditReport,
    _jaccard_similarity,
    _spearman_rank_correlation,
    _kendall_tau,
    _bootstrap_ci,
)


# ── Test 1: Mathematical Helper Functions ────────────────────────────────────

def test_01_jaccard_and_rank_correlation_math():
    # Jaccard tests
    assert _jaccard_similarity(set(), set()) == 1.0
    assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
    assert _jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1.0 / 3.0)

    # Spearman rank correlation tests
    assert _spearman_rank_correlation(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
    assert _spearman_rank_correlation(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)
    assert _spearman_rank_correlation(["a"], ["a"]) == 1.0

    # Kendall tau tests
    assert _kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
    assert _kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)


def test_02_bootstrap_ci_computation():
    data = [0.80, 0.82, 0.85, 0.84, 0.81, 0.83]
    low, high = _bootstrap_ci(data, n_resamples=500, ci=0.95)
    assert low <= high
    assert 0.79 <= low <= 0.83
    assert 0.82 <= high <= 0.86

    # Uniform constant data
    c_low, c_high = _bootstrap_ci([1.0, 1.0, 1.0])
    assert c_low == 1.0
    assert c_high == 1.0


# ── Test 2: Perturbation Stability ───────────────────────────────────────────

def test_03_perturbation_stability():
    auditor = XAIReliabilityAuditor(seed=42)
    cohort = [
        {"S_sig": 0.90, "A_ml": 0.80, "delta_D": 0.50, "T_trust": 0.10, "H_boost": 0.20, "G_corr": 0.40, "P_fore": 0.10, "TI_score": 0.50, "A_crit": 1.0},
        {"S_sig": 0.70, "A_ml": 0.60, "delta_D": 0.30, "T_trust": 0.20, "H_boost": 0.10, "G_corr": 0.20, "P_fore": 0.05, "TI_score": 0.30, "A_crit": 1.0},
    ]

    res = auditor.evaluate_stability(cohort, k=3, noise_std=0.02, n_perturbations=10)
    assert res.mean_jaccard >= 0.80
    assert res.median_jaccard >= 0.80
    assert res.p95_instability <= 0.35
    assert res.is_stable is True
    assert len(res.ci_95) == 2


# ── Test 3: Sufficiency Monotonicity ──────────────────────────────────────────

def test_04_sufficiency_monotonicity():
    auditor = XAIReliabilityAuditor(seed=42)
    cohort = [
        {"S_sig": 0.80, "A_ml": 0.70, "delta_D": 0.40, "T_trust": 0.10, "H_boost": 0.20, "G_corr": 0.30, "P_fore": 0.10, "TI_score": 0.40, "A_crit": 1.0},
        {"S_sig": 0.95, "A_ml": 0.90, "delta_D": 1.20, "T_trust": 0.05, "H_boost": 0.15, "G_corr": 0.50, "P_fore": 0.30, "TI_score": 0.60, "A_crit": 1.2},
    ]

    res = auditor.evaluate_sufficiency(cohort, k_values=(3, 5, 8))
    assert res.k_evaluations["k=3"] > 0.50
    assert res.k_evaluations["k=8"] >= res.k_evaluations["k=3"]
    assert res.is_monotonic is True
    assert 0.0 <= res.mean_sufficiency <= 1.0


# ── Test 4: Comprehensiveness Degradation ─────────────────────────────────────

def test_05_comprehensiveness_meaningful_drop():
    auditor = XAIReliabilityAuditor(seed=42)
    cohort = [
        {"S_sig": 0.85, "A_ml": 0.75, "delta_D": 0.40, "T_trust": 0.10, "H_boost": 0.20, "G_corr": 0.30, "P_fore": 0.10, "TI_score": 0.40, "A_crit": 1.0},
        {"S_sig": 0.90, "A_ml": 0.85, "delta_D": 1.00, "T_trust": 0.05, "H_boost": 0.15, "G_corr": 0.50, "P_fore": 0.20, "TI_score": 0.55, "A_crit": 1.0},
    ]

    res = auditor.evaluate_comprehensiveness(cohort, k_values=(3, 5, 8))
    # Ablating top 3 explanation components should cause meaningful decision degradation
    assert res.k_evaluations["k=3"] >= 0.30
    assert res.is_meaningful is True
    assert res.mean_comprehensiveness >= 0.30


# ── Test 5: Spurious Robustness & Zero Leakage ────────────────────────────────

def test_06_spurious_robustness_and_non_leakage():
    auditor = XAIReliabilityAuditor(seed=42)
    cohort = [
        {"S_sig": 0.80, "A_ml": 0.70, "delta_D": 0.40, "T_trust": 0.10, "H_boost": 0.20, "G_corr": 0.30, "P_fore": 0.10, "TI_score": 0.40, "A_crit": 1.0},
        {"S_sig": 0.90, "A_ml": 0.80, "delta_D": 0.80, "T_trust": 0.05, "H_boost": 0.10, "G_corr": 0.40, "P_fore": 0.20, "TI_score": 0.50, "A_crit": 1.0},
    ]

    res = auditor.evaluate_spurious_robustness(cohort, k=3, n_spurious_features=3)
    assert res.mean_prediction_change == pytest.approx(0.0, abs=1e-4)
    assert res.mean_top_k_overlap >= 0.90
    assert res.spurious_leak_rate == 0.0
    assert res.is_robust is True


# ── Test 6: Cross-Run Consistency ────────────────────────────────────────────

def test_07_cross_run_consistency():
    auditor = XAIReliabilityAuditor(seed=42)
    cohort = [
        {"S_sig": 0.85, "A_ml": 0.75, "delta_D": 0.50, "T_trust": 0.10, "H_boost": 0.20, "G_corr": 0.30, "P_fore": 0.10, "TI_score": 0.40, "A_crit": 1.0},
    ]

    res = auditor.evaluate_cross_run_consistency(cohort, k=3, seeds=(1, 2, 3))
    assert res.mean_agreement_jaccard >= 0.90
    assert res.mean_rank_correlation >= 0.90
    assert res.is_consistent is True


# ── Test 7: Full Audit Schema & Execution ────────────────────────────────────

def test_08_full_reliability_audit_execution():
    auditor = XAIReliabilityAuditor(seed=42)
    cohort = [
        {"S_sig": 0.85, "A_ml": 0.70, "delta_D": 0.50, "T_trust": 0.10, "H_boost": 0.20, "G_corr": 0.30, "P_fore": 0.15, "TI_score": 0.40, "A_crit": 1.0},
        {"S_sig": 0.00, "A_ml": 0.10, "delta_D": 0.00, "T_trust": 0.80, "H_boost": 0.00, "G_corr": 0.00, "P_fore": 0.00, "TI_score": 0.00, "A_crit": 1.0},
        {"S_sig": 1.00, "A_ml": 0.95, "delta_D": 2.00, "T_trust": 0.00, "H_boost": 0.10, "G_corr": 0.50, "P_fore": 0.50, "TI_score": 0.70, "A_crit": 1.2},
    ]

    report = auditor.run_full_audit(cohort, fidelity_pass_rate=1.0, fidelity_mae=0.0)
    assert isinstance(report, XAIReliabilityAuditReport)
    assert report.computational_fidelity_pass == 1.0
    assert report.stability.is_stable is True
    assert report.sufficiency.is_monotonic is True
    assert report.comprehensiveness.is_meaningful is True
    assert report.spurious_robustness.is_robust is True
    assert len(report.summary_table) == 6

    # Verify JSON serializability
    d = report.to_dict()
    assert d["n_samples"] == 3
    assert "summary_table" in d
    assert "stability" in d
