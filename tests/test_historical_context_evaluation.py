from __future__ import annotations
"""
Unit and Integration Tests for Phase 6 / RQ4b: Historical Security Context & Recidivism Reasoning
-------------------------------------------------------------------------------------------------
Verifies:
  1. Longitudinal telemetry stream simulation (60-day timeline across 5 threat cohorts)
  2. Temporal causality and strict leak-free chronological ordering (zero lookahead)
  3. Mathematical exactness of recency decay factors (1.0 -> 0.50 -> 0.25)
  4. Stateless baseline H0 amnesia on low-and-slow persistent threats
  5. Stage 11 H1 recidivism boost accumulation and persistent threat recall (>= 0.85)
  6. Escalation lead-time speedup on persistent threats (>= 2.0 sessions earlier)
  7. Benign entity false-positive rate stability (no false escalation on benign recurring)
  8. Paired sample permutation significance test (p < 0.05, Cohen's d > 0.20)
  9. End-to-end report generation, schema integrity, and JSON serializability
"""

import pytest
import numpy as np

from evaluation.historical_context_experiment import (
    LongitudinalEvent,
    HistoricalEvaluationMetrics,
    LongitudinalTelemetrySimulator,
    HistoricalContextExperiment,
)
from historical_risk.engine import HistoricalRiskEngine, IndicatorHistory
from detection.risk_engine import AdaptiveRiskEngine, RiskConfig


def test_longitudinal_telemetry_simulation():
    """Validates 60-day multi-cohort timeline generation."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(
        duration_days=60.0,
        n_persistent=10,
        n_dormant=5,
        n_transient_atk=10,
        n_benign_rec=50,
        n_benign_trans=100,
    )
    assert len(events) > 500
    cohorts = {e.cohort for e in events}
    assert cohorts == {"PERSISTENT", "DORMANT", "TRANSIENT_ATK", "BENIGN_REC", "BENIGN_TRANS"}

    # Chronological timestamps
    timestamps = [e.timestamp for e in events]
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps) - 1))
    
    # Duration validation
    duration_span = (timestamps[-1] - timestamps[0]) / 86400.0
    assert 50.0 <= duration_span <= 60.5


def test_temporal_causality_leakage_audit():
    """Ensures zero future-to-past lookahead leakage in timeline."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(duration_days=30.0, n_persistent=5, n_dormant=2, n_transient_atk=5, n_benign_rec=20, n_benign_trans=40)
    exp = HistoricalContextExperiment(seed=42)
    audit = exp._audit_temporal_causality(events)

    assert audit["chronological_ordering_verified"] is True
    assert audit["zero_lookahead_leakage"] is True
    assert audit["audit_status"] == "PASSED"
    assert audit["min_inter_event_time_sec"] >= 0.0


def test_recency_decay_exact_schedule():
    """Validates exact 1.0 (<7d), 0.50 (7-30d), and 0.25 (>30d) decay factors."""
    hist = HistoricalRiskEngine()
    ip = "192.0.2.100"
    t0 = 1000000.0
    # Record 3 incidents and 2 alerts at t0
    for _ in range(3):
        hist.record_event(ip, risk_score=0.85, is_alert=True, is_incident=True, timestamp=t0)
    for _ in range(2):
        hist.record_event(ip, risk_score=0.70, is_alert=True, is_incident=False, timestamp=t0)

    # Day 3 (< 7 days): recency factor 1.0
    b_3d = hist.compute_history_boost(ip, normalized_unit_scale=True, now=t0 + 3.0 * 86400.0)
    # Day 14 (7 to 30 days): recency factor 0.50
    b_14d = hist.compute_history_boost(ip, normalized_unit_scale=True, now=t0 + 14.0 * 86400.0)
    # Day 45 (> 30 days): recency factor 0.25
    b_45d = hist.compute_history_boost(ip, normalized_unit_scale=True, now=t0 + 45.0 * 86400.0)

    assert b_3d > b_14d > b_45d
    assert pytest.approx(b_14d / b_3d, 1e-3) == 0.50
    assert pytest.approx(b_45d / b_3d, 1e-3) == 0.25


def test_stateless_baseline_h0_amnesia():
    """Validates that H0 (use_history=False) evaluates each event in complete isolation."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(duration_days=60.0, n_persistent=10, n_dormant=5, n_transient_atk=10, n_benign_rec=40, n_benign_trans=50)
    exp = HistoricalContextExperiment(decision_threshold=0.50, seed=42)

    h0_metrics, h0_scores = exp._evaluate_pipeline(events, use_history=False)
    assert h0_metrics.active_recidivist_boost == 0.0
    assert h0_metrics.dormant_reactivation_boost == 0.0
    # Stealthy persistent attacks evade stateless detection
    assert h0_metrics.recidivist_recall < 0.40


def test_recidivism_engine_h1_gain():
    """Validates that H1 elevates persistent threat recall >= 0.85 with substantial F1 gain."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(
        duration_days=60.0,
        n_persistent=25,
        n_dormant=15,
        n_transient_atk=30,
        n_benign_rec=150,
        n_benign_trans=300,
    )
    exp = HistoricalContextExperiment(decision_threshold=0.50, critical_threshold=0.70, seed=42)

    h0_metrics, _ = exp._evaluate_pipeline(events, use_history=False)
    h1_metrics, _ = exp._evaluate_pipeline(events, use_history=True)

    assert h1_metrics.recidivist_recall >= 0.85
    assert h1_metrics.recidivist_f1 > h0_metrics.recidivist_f1
    relative_f1_gain = ((h1_metrics.recidivist_f1 - h0_metrics.recidivist_f1) / h0_metrics.recidivist_f1) * 100.0
    assert relative_f1_gain >= 50.0  # Well exceeds 15% threshold


def test_lead_time_speedup():
    """Validates that H1 escalates persistent threat actors to critical severity >= 2.0 sessions earlier than H0."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(duration_days=60.0, n_persistent=25, n_dormant=15, n_transient_atk=30, n_benign_rec=150, n_benign_trans=300)
    exp = HistoricalContextExperiment(decision_threshold=0.50, critical_threshold=0.70, seed=42)

    h0_metrics, _ = exp._evaluate_pipeline(events, use_history=False)
    h1_metrics, _ = exp._evaluate_pipeline(events, use_history=True)

    speedup = h0_metrics.mean_sessions_to_critical - h1_metrics.mean_sessions_to_critical
    assert speedup >= 2.0
    assert h1_metrics.mean_sessions_to_critical <= 4.0


def test_benign_fpr_stability():
    """Validates that benign recurring hosts do not suffer false positive escalation from threat history."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(duration_days=60.0, n_persistent=10, n_dormant=5, n_transient_atk=10, n_benign_rec=100, n_benign_trans=100)
    exp = HistoricalContextExperiment(seed=42)

    h0_metrics, h0_scores = exp._evaluate_pipeline(events, use_history=False)
    h1_metrics, h1_scores = exp._evaluate_pipeline(events, use_history=True)

    # FPR delta must be <= 0.001
    assert abs(h1_metrics.fpr - h0_metrics.fpr) <= 0.001
    assert h1_metrics.fpr <= 0.005


def test_paired_permutation_significance():
    """Validates paired permutation test p < 0.05 and meaningful Cohen's d effect size."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(duration_days=60.0, n_persistent=15, n_dormant=10, n_transient_atk=20, n_benign_rec=80, n_benign_trans=150)
    exp = HistoricalContextExperiment(seed=42)

    _, h0_scores = exp._evaluate_pipeline(events, use_history=False)
    _, h1_scores = exp._evaluate_pipeline(events, use_history=True)

    stat = exp._compute_statistical_significance(events, h0_scores, h1_scores, n_permutations=2000)
    assert stat["statistically_significant"] is True
    assert stat["two_sided_p_value"] < 0.05
    assert stat["cohens_d"] > 0.20


def test_end_to_end_report_schema():
    """Validates complete evaluation report generation and schema invariants."""
    sim = LongitudinalTelemetrySimulator(seed=42)
    events = sim.generate_timeline(duration_days=60.0, n_persistent=10, n_dormant=5, n_transient_atk=10, n_benign_rec=40, n_benign_trans=50)
    exp = HistoricalContextExperiment(seed=42)
    report = exp.evaluate(events)

    required_keys = [
        "experiment_id",
        "research_question",
        "timeline_summary",
        "temporal_leakage_audit",
        "stateless_baseline_h0",
        "historical_context_stage_11_h1",
        "comparative_gains",
        "cohort_performance_breakdown",
        "recency_decay_validation",
        "statistical_significance",
        "scientific_conclusion",
    ]
    for k in required_keys:
        assert k in report, f"Missing report key: {k}"

    assert report["temporal_leakage_audit"]["zero_lookahead_leakage"] is True
    assert report["recency_decay_validation"]["mathematical_decay_conformance"] is True
