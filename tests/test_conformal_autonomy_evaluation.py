from __future__ import annotations
"""
AHRAS Test Suite — Phase 10 / RQ6 Safe Selective Autonomy & Conformal Risk Gating
----------------------------------------------------------------------------------
Comprehensive unit and integration test suite validating:
  1. ConformalRiskGate split calibration and quantile nonconformity tau* bounds.
  2. 7-tier operational action arbitration (AUTONOMOUS_PASS, MONITOR, DECEPTION,
     STAGED_CONTAINMENT, AUTONOMOUS_CONTAINMENT, ABSTAIN, ESCALATE_ANALYST).
  3. Risk-to-Action Safety Efficiency (RASE) mathematical properties.
  4. 1,000 incident scenarios generator distribution and provenance.
  5. Policy evaluation matrix and false intervention reduction (>= 65%).
  6. False autonomous containment elimination (>= 75%).
  7. RASE safety efficiency improvement (>= 35%).
  8. Paired sample permutation test and Cohen's d effect size.
  9. CLM-06 claims manifest consistency with CONFORMAL_AUTONOMY_REPORT.json.
"""

import os
import sys
import json
import pytest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detection.selective_gate import (
    ConformalRiskGate,
    SelectionDecision,
    ACTION_AUTONOMOUS_PASS,
    ACTION_MONITOR,
    ACTION_DECEPTION,
    ACTION_STAGED_CONTAINMENT,
    ACTION_AUTONOMOUS_CONTAINMENT,
    ACTION_ABSTAIN,
    ACTION_ESCALATE_ANALYST,
)
from detection.risk_engine import compute_rase
from evaluation.conformal_autonomy_experiment import (
    ConformalAutonomyExperiment,
    paired_permutation_test,
)


def test_conformal_gate_calibration_properties():
    """Verifies that split conformal calibration computes a valid nonconformity quantile tau*."""
    gate = ConformalRiskGate(target_coverage=0.90)
    risk_scores = [0.05, 0.12, 0.18, 0.85, 0.92, 0.88, 0.22, 0.79]
    labels = [0, 0, 0, 1, 1, 1, 0, 1]

    tau = gate.calibrate(risk_scores, labels)
    assert gate.is_calibrated is True
    assert 0.0 < tau <= 1.0
    assert gate.alpha == pytest.approx(0.10, abs=1e-4)


def test_seven_tier_operational_arbitration():
    """Verifies that all 7 operational defense tiers are triggered under expected conditions."""
    gate = ConformalRiskGate(target_coverage=0.90, uncertainty_threshold=0.35, risk_action_threshold=0.70)
    gate.calibrated_tau = 0.25

    # 1. Low risk, low uncertainty -> AUTONOMOUS_PASS
    d1 = gate.evaluate_gate(risk_score=0.10, uncertainty=0.05, ood_score=0.05)
    assert d1.action == ACTION_AUTONOMOUS_PASS
    assert d1.is_autonomous is True

    # 2. Moderate risk -> MONITOR
    d2 = gate.evaluate_gate(risk_score=0.50, uncertainty=0.15, ood_score=0.10)
    assert d2.action == ACTION_MONITOR
    assert d2.is_autonomous is True

    # 3. High OOD zero-day -> DECEPTION
    d3 = gate.evaluate_gate(risk_score=0.60, uncertainty=0.20, ood_score=0.85)
    assert d3.action == ACTION_DECEPTION
    assert d3.is_autonomous is True

    # 4. High confidence critical attack -> AUTONOMOUS_CONTAINMENT
    d4 = gate.evaluate_gate(risk_score=0.95, uncertainty=0.10, ood_score=0.10)
    assert d4.action == ACTION_AUTONOMOUS_CONTAINMENT
    assert d4.is_autonomous is True

    # 5. Elevated risk under borderline conditions -> STAGED_CONTAINMENT
    d5 = gate.evaluate_gate(risk_score=0.75, uncertainty=0.25, ood_score=0.20)
    assert d5.action == ACTION_STAGED_CONTAINMENT
    assert d5.is_autonomous is True

    # 6. Ambiguous near-threshold risk with high nonconformity -> ABSTAIN
    d6 = gate.evaluate_gate(risk_score=0.68, uncertainty=0.45, ood_score=0.20)
    assert d6.action == ACTION_ABSTAIN
    assert d6.is_autonomous is False

    # 7. High risk with epistemic disagreement -> ESCALATE_ANALYST
    d7 = gate.evaluate_gate(risk_score=0.85, uncertainty=0.60, ood_score=0.20)
    assert d7.action == ACTION_ESCALATE_ANALYST
    assert d7.is_autonomous is False


def test_rase_metric_mathematical_properties():
    """Verifies that RASE metric strictly scales with risk reduction and penalizes blast radius and uncertainty."""
    # Base surgical containment
    rase_surgical = compute_rase(
        risk_reduction=0.80,
        uncertainty=0.10,
        blast_radius=0.15,
        reversibility_cost=0.10,
        is_false_intervention=False,
    )

    # Blunt uncalibrated containment
    rase_blunt = compute_rase(
        risk_reduction=0.80,
        uncertainty=0.50,
        blast_radius=0.70,
        reversibility_cost=0.30,
        is_false_intervention=False,
    )

    assert rase_surgical > rase_blunt * 2.0, "Surgical containment must achieve >2x safety efficiency over blunt isolation"

    # False intervention penalty
    rase_fp = compute_rase(
        risk_reduction=0.80,
        uncertainty=0.10,
        blast_radius=0.15,
        reversibility_cost=0.10,
        is_false_intervention=True,
        lambda_fp_penalty=2.0,
    )
    assert rase_fp < rase_surgical


def test_scenario_generation_distribution():
    """Verifies the scenario synthesis distribution (40% attacks, 60% benign, noisy boundary cases)."""
    exp = ConformalAutonomyExperiment(seed=42)
    calib, test_scenarios = exp.generate_incident_scenarios(n_scenarios=500, n_calibration=100)

    assert len(calib) == 100
    assert len(test_scenarios) == 500

    attacks = [s for s in test_scenarios if s.is_attack == 1]
    benign = [s for s in test_scenarios if s.is_attack == 0]
    assert len(attacks) == 200  # 40%
    assert len(benign) == 300   # 60%

    # Verify presence of noisy boundary events (R >= 0.70 with high uncertainty)
    noisy_benign = [s for s in benign if s.risk_score >= 0.70 and s.uncertainty >= 0.35]
    assert len(noisy_benign) > 30, "Scenario generator must inject noisy boundary cases testing false containment"


def test_conformal_autonomy_experiment_execution():
    """Runs ConformalAutonomyExperiment and validates output schema and policy metrics."""
    exp = ConformalAutonomyExperiment(seed=42)
    report = exp.run_evaluation(n_scenarios=200, n_calibration=50, target_coverage=0.90)

    assert report["experiment_id"] == "EXP-06"
    assert report["research_question"] == "RQ6: Safe Selective Autonomy & Conformal Risk Gating"
    assert "provenance" in report
    assert "policies_evaluated" in report
    assert "statistical_significance_vs_uncalibrated" in report
    assert "hypothesis_verification" in report
    assert "claims_manifest_entry" in report

    policies = {p["policy_name"]: p for p in report["policies_evaluated"]}
    assert "Policy_Uncalibrated_Fixed_Threshold" in policies
    assert "Policy_AHRAS_Conformal_Risk_Gate" in policies

    ahras = policies["Policy_AHRAS_Conformal_Risk_Gate"]
    uncal = policies["Policy_Uncalibrated_Fixed_Threshold"]

    assert ahras["attack_containment_rate"] >= 90.0
    assert ahras["false_intervention_rate"] <= uncal["false_intervention_rate"]
    assert ahras["mean_rase_safety_score"] > uncal["mean_rase_safety_score"]


def test_false_intervention_and_autonomous_containment_reduction():
    """Verifies core theoretical invariants: >= 65% false intervention reduction and >= 75% false containment reduction."""
    exp = ConformalAutonomyExperiment(seed=42)
    report = exp.run_evaluation(n_scenarios=500, n_calibration=150, target_coverage=0.90)

    hyp = report["hypothesis_verification"]
    assert hyp["false_intervention_reduction_reaches_65_pct"] is True
    assert hyp["false_autonomous_containment_reduction_reaches_75_pct"] is True
    assert hyp["attack_containment_reaches_90_pct"] is True


def test_rase_improvement_and_statistical_significance():
    """Verifies RASE score improvement >= 35% and paired permutation statistical significance."""
    exp = ConformalAutonomyExperiment(seed=42)
    report = exp.run_evaluation(n_scenarios=500, n_calibration=150, target_coverage=0.90)

    hyp = report["hypothesis_verification"]
    assert hyp["rase_improvement_reaches_35_pct"] is True

    stat = report["statistical_significance_vs_uncalibrated"]
    assert stat["statistically_significant"] is True
    assert stat["p_value"] < 0.05
    assert stat["mean_rase_gain"] > 0.10


def test_claims_manifest_clm06_consistency():
    """Verifies that CLAIMS_MANIFEST_FINAL.json CLM-06 is supported and consistent."""
    claims_path = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")
    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    assert "CLM-06" in claims
    clm = claims["CLM-06"]
    assert clm["status"] == "SUPPORTED"
    assert clm["metric"] == "rase_safety_score"
    assert clm["value"] >= 0.40
    assert clm["false_intervention_reduction_pct"] >= 65.0
    assert clm["rase_improvement_pct"] >= 35.0
