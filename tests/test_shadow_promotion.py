"""
Tests for Shadow Model Promotion & Safety Pipeline (Section 25 / Research Frontier P1)
----------------------------------------------------------------------------------------
Validates:
  1. Registration of baseline champion model.
  2. Candidate enrollment into shadow evaluation state.
  3. Strict evaluation of all 9 promotion gates:
     - F1 non-degradation
     - FPR increase control
     - Unknown OOD recall
     - Calibration ECE
     - XAI rank stability
     - P95 latency SLA
     - Memory growth ratio
     - ATT&CK coverage preservation
     - Drift stability
  4. Automatic promotion of superior candidate to new champion.
  5. Deterministic rejection when any gate fails (e.g. latency breach, calibration failure).
"""

import pytest
from adaptive_learning.shadow_promotion import (
    ModelEvaluationMetrics,
    PromotionGateCheck,
    PromotionDecision,
    PromotionPolicy,
    PromotionStatus,
    ShadowModelPromotionPipeline,
)


@pytest.fixture
def champion_metrics():
    return ModelEvaluationMetrics(
        model_name="HybridEnsemble",
        version="v1.0",
        f1_score=0.965,
        precision=0.970,
        recall=0.960,
        fpr=0.012,
        unknown_ood_recall=0.880,
        ece_calibration=0.075,
        brier_score=0.088,
        xai_stability_jaccard=0.820,
        p50_latency_ms=2.10,
        p95_latency_ms=12.50,
        memory_mb=45.0,
        cpu_cost_factor=1.0,
        mitre_technique_coverage=23,
        drift_statistic=12.4,
    )


def test_champion_registration_and_enrollment(champion_metrics):
    pipeline = ShadowModelPromotionPipeline()
    pipeline.register_champion(champion_metrics)

    candidate = ModelEvaluationMetrics(
        model_name="HybridEnsemble",
        version="v1.1-cand",
        f1_score=0.972,
        precision=0.975,
        recall=0.969,
        fpr=0.010,
        unknown_ood_recall=0.910,
        ece_calibration=0.062,
        brier_score=0.075,
        xai_stability_jaccard=0.850,
        p50_latency_ms=1.95,
        p95_latency_ms=11.20,
        memory_mb=48.0,
        cpu_cost_factor=1.05,
        mitre_technique_coverage=27,
        drift_statistic=9.8,
    )
    pipeline.enroll_shadow_candidate(candidate)

    decision = pipeline.evaluate_promotion("HybridEnsemble:v1.1-cand")
    assert decision.all_gates_passed is True
    assert decision.passed_gate_count == 9
    assert decision.status == PromotionStatus.PROMOTED
    assert len(decision.rejection_reasons) == 0


def test_candidate_rejected_on_latency_breach(champion_metrics):
    pipeline = ShadowModelPromotionPipeline(policy=PromotionPolicy(max_p95_latency_ms=25.0))
    pipeline.register_champion(champion_metrics)

    # Slow candidate that breaches 25ms SLA
    slow_candidate = ModelEvaluationMetrics(
        model_name="HybridEnsemble",
        version="v1.2-slow",
        f1_score=0.980,  # Higher F1
        precision=0.980,
        recall=0.980,
        fpr=0.008,
        unknown_ood_recall=0.920,
        ece_calibration=0.050,
        brier_score=0.060,
        xai_stability_jaccard=0.860,
        p50_latency_ms=15.0,
        p95_latency_ms=32.4,  # BREACH!
        memory_mb=50.0,
        cpu_cost_factor=2.0,
        mitre_technique_coverage=25,
        drift_statistic=10.0,
    )
    pipeline.enroll_shadow_candidate(slow_candidate)

    decision = pipeline.evaluate_promotion("HybridEnsemble:v1.2-slow")
    assert decision.all_gates_passed is False
    assert decision.status == PromotionStatus.REJECTED
    assert any("P95 latency SLA breached" in r for r in decision.rejection_reasons)


def test_candidate_rejected_on_calibration_degradation(champion_metrics):
    pipeline = ShadowModelPromotionPipeline(policy=PromotionPolicy(max_allowed_ece=0.15))
    pipeline.register_champion(champion_metrics)

    # Uncalibrated candidate with high ECE
    uncalibrated_candidate = ModelEvaluationMetrics(
        model_name="HybridEnsemble",
        version="v1.3-uncal",
        f1_score=0.970,
        precision=0.970,
        recall=0.970,
        fpr=0.015,
        unknown_ood_recall=0.890,
        ece_calibration=0.220,  # Uncalibrated (> 0.15)
        brier_score=0.180,
        xai_stability_jaccard=0.800,
        p50_latency_ms=2.0,
        p95_latency_ms=12.0,
        memory_mb=45.0,
        cpu_cost_factor=1.0,
        mitre_technique_coverage=23,
        drift_statistic=11.0,
    )
    pipeline.enroll_shadow_candidate(uncalibrated_candidate)

    decision = pipeline.evaluate_promotion("HybridEnsemble:v1.3-uncal")
    assert decision.all_gates_passed is False
    assert any("Calibration ECE too high" in r for r in decision.rejection_reasons)


def test_candidate_rejected_on_coverage_decrease(champion_metrics):
    pipeline = ShadowModelPromotionPipeline(policy=PromotionPolicy(require_coverage_non_decreasing=True))
    pipeline.register_champion(champion_metrics)

    # Candidate with decreased coverage
    reduced_coverage_candidate = ModelEvaluationMetrics(
        model_name="HybridEnsemble",
        version="v1.4-lowcov",
        f1_score=0.970,
        precision=0.970,
        recall=0.970,
        fpr=0.010,
        unknown_ood_recall=0.890,
        ece_calibration=0.070,
        brier_score=0.080,
        xai_stability_jaccard=0.810,
        p50_latency_ms=2.0,
        p95_latency_ms=12.0,
        memory_mb=45.0,
        cpu_cost_factor=1.0,
        mitre_technique_coverage=18,  # Dropped from 23
        drift_statistic=10.0,
    )
    pipeline.enroll_shadow_candidate(reduced_coverage_candidate)

    decision = pipeline.evaluate_promotion("HybridEnsemble:v1.4-lowcov")
    assert decision.all_gates_passed is False
    assert any("MITRE coverage decreased" in r for r in decision.rejection_reasons)
