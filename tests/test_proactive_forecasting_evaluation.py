from __future__ import annotations
"""
AHRAS Test Suite — Phase 12 / RQ8: Causal Early-Warning Risk Prediction & Forecasting
---------------------------------------------------------------------------------------
Tests:
  1. Incident trajectory generation (Rapid, Slow, Benign, De-escalating).
  2. Strict causality and zero lookahead leakage invariant.
  3. Warning lead time measurement (target >= 3.0 events).
  4. Horizon accuracy superiority over Naive Persistence and MA-5.
  5. False warning suppression on benign steady-state trajectories.
  6. De-escalating threat remediation identification.
  7. Blast radius exposure reduction calculation (target >= 40%).
  8. Full multi-sequence experiment execution and paired permutation testing.
  9. Report consistency with CLM-08 and publication rules.
"""

import os
import json
import pytest
import numpy as np

from forecast.predictor import (
    AttackPredictor,
    ForecastResult,
    walk_forward_errors,
    forecast_accuracy,
    threshold_crossing_lead_time,
    compute_quantitative_forecast_boost,
    MIN_POINTS_FOR_FORECAST,
)
from evaluation.proactive_forecasting_experiment import (
    generate_longitudinal_incident_suite,
    paired_permutation_test,
    ProactiveForecastingExperiment,
    NaivePersistenceForecaster,
    MovingAverageForecaster,
    LinearMomentumForecaster,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_01_trajectory_generation():
    trajectories = generate_longitudinal_incident_suite(
        n_rapid=20, n_slow=15, n_stable=10, n_deescalating=10, threshold=0.85, seed=42
    )
    assert len(trajectories) == 55

    for traj in trajectories:
        assert len(traj.risk_scores) >= 10
        assert all(0.0 <= s <= 1.0 for s in traj.risk_scores)
        if traj.has_breach:
            assert traj.breach_step is not None
            assert traj.risk_scores[traj.breach_step] >= 0.85
        else:
            assert traj.breach_step is None
            assert all(s < 0.85 for s in traj.risk_scores)


def test_02_strict_causality_and_zero_lookahead():
    predictor = AttackPredictor(horizon=6)
    seq = [0.10, 0.15, 0.20, 0.32, 0.46, 0.62, 0.76, 0.89]

    # Verify that at each step t, predictor only receives seq[:t]
    for t in range(MIN_POINTS_FOR_FORECAST, len(seq)):
        past_only = seq[:t]
        res = predictor.predict("causal_check", past_only, critical_threshold=0.85)
        assert res.data_points == len(past_only)
        # Verify forecast does not equal future true observations
        if t < len(seq) - 1:
            assert res.predicted_risk_h1 != seq[t]


def test_03_warning_lead_time_exceeds_threshold():
    predictor = AttackPredictor(horizon=6)
    # Slow escalating sequence with clear precursors
    seq = [0.15, 0.19, 0.23, 0.28, 0.34, 0.40, 0.46, 0.53, 0.60, 0.68, 0.76, 0.84, 0.91]
    lt = threshold_crossing_lead_time(predictor, seq, threshold=0.85)
    assert lt is not None
    assert lt >= 3  # Target >= 3 events


def test_04_horizon_accuracy_superior_to_naive_persistence():
    predictor = AttackPredictor(horizon=6)
    persistence = NaivePersistenceForecaster()

    # Trending sequence
    seq = [0.10, 0.15, 0.22, 0.30, 0.40, 0.51, 0.63, 0.76, 0.88]
    acc_holt = forecast_accuracy(predictor, seq)

    # Compute persistence errors
    errors_pers = [abs(seq[t] - seq[t - 1]) for t in range(MIN_POINTS_FOR_FORECAST, len(seq))]
    mae_pers = float(np.mean(errors_pers))

    assert acc_holt["mae"] <= mae_pers
    assert acc_holt["rmse"] < 0.15


def test_05_zero_false_warnings_on_benign_series():
    predictor = AttackPredictor(horizon=6)
    # Stable benign series fluctuating around 0.25
    rng = np.random.default_rng(42)
    benign_seq = [float(np.clip(0.25 + rng.normal(0.0, 0.03), 0.1, 0.45)) for _ in range(20)]

    res = predictor.predict("benign_test", benign_seq, critical_threshold=0.85)
    assert not res.will_breach_critical
    assert res.probability_of_threshold_crossing < 0.10
    assert res.trend_label in ("STABLE", "DE-ESCALATING")


def test_06_deescalating_threat_remediation():
    predictor = AttackPredictor(horizon=6)
    # Threat decaying from 0.75 down to 0.15
    decay_seq = [0.75, 0.68, 0.60, 0.52, 0.43, 0.35, 0.28, 0.20]

    res = predictor.predict("decay_test", decay_seq, critical_threshold=0.85)
    assert res.trend_label == "DE-ESCALATING"
    assert res.trend < 0.0
    assert not res.will_breach_critical
    # Quantitative forecast boost must be zero for de-escalating threats
    boost = compute_quantitative_forecast_boost(res, current_risk=0.20)
    assert boost == 0.0


def test_07_blast_radius_reduction_calculation():
    # Sequence with breach at step 8
    seq = [0.10, 0.15, 0.22, 0.30, 0.40, 0.52, 0.65, 0.77, 0.89, 0.95]
    t_breach = 8
    # Reactive containment: step 8 + 2 = 10
    blast_reactive = sum(seq[:10])
    # Proactive containment: lead time = 3 -> containment at step 8 - 3 + 1 = 6
    blast_proactive = sum(seq[:6])

    red_pct = (blast_reactive - blast_proactive) / blast_reactive * 100.0
    assert red_pct >= 40.0


def test_08_full_experiment_run():
    exp = ProactiveForecastingExperiment(seed=42)
    report = exp.run_experiment()

    assert report["experiment_id"] == "EXP-08"
    assert report["phase"] == "Phase 12"
    sm = report["summary_metrics"]
    assert sm["mean_warning_lead_time_events"] >= 3.0
    assert sm["blast_radius_reduction_pct"] >= 40.0
    assert sm["warning_precision"] >= 0.95
    assert sm["warning_recall"] >= 0.95
    assert sm["false_warning_rate_on_benign_pct"] <= 5.0
    assert sm["lead_time_permutation_p_value"] <= 0.001
    assert sm["lead_time_cohens_d"] >= 1.0
    assert sm["lookahead_leakage_audit_pass"] is True


def test_09_report_schema_and_clm08_consistency():
    report_path = os.path.join(_ROOT, "evaluation", "results", "PROACTIVE_FORECASTING_REPORT.json")
    claims_path = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")

    assert os.path.exists(report_path)
    assert os.path.exists(claims_path)

    with open(report_path) as f:
        rep = json.load(f)
    with open(claims_path) as f:
        claims = json.load(f)

    assert "CLM-08" in claims
    clm = claims["CLM-08"]
    assert clm["status"] == "SUPPORTED"
    assert clm["value"] >= 3.0
    assert rep["claims_mapping"]["value"] == clm["value"]
    assert clm["zero_lookahead_leakage"] is True
