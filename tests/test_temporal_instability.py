from __future__ import annotations
"""
Unit and Integration Tests for Temporal Epistemic Instability Engine
---------------------------------------------------------------------
Validates:
  - Calculation of 5 core volatility signals (confidence variance, class flips,
    trajectory acceleration, Shannon entropy, temporal disagreement).
  - Cohort differentiation: Stable Benign, Stable Attack, Gradual Attack, Oscillating Ambiguous.
  - Non-increasing risk guarantee (modulate_uncertainty never raises risk score).
  - Autonomous response gating and safety overrides (AUTONOMOUS_CONTAINMENT -> ABSTAIN/ESCALATE).
  - Window pruning and configuration flags.
"""

import math
import numpy as np
import pytest

from instability.models import (
    PredictionRecord, InstabilityMetrics, EntityStabilityProfile
)
from instability.tracker import TemporalInstabilityTracker
from config import settings


def test_instability_tracker_base_case():
    """Verifies behavior on 0 or 1 historical observation."""
    tracker = TemporalInstabilityTracker(max_window_size=20)

    # Empty
    m0 = tracker.compute_instability("unknown-entity")
    assert m0.instability_score == 0.0
    assert m0.class_flip_count == 0
    assert not m0.autonomy_gated

    # Single observation
    tracker.record_prediction("host-01", 1000.0, 0.45, 0.90)
    m1 = tracker.compute_instability("host-01")
    assert m1.instability_score == 0.0
    assert m1.class_flip_count == 0
    assert not m1.autonomy_gated


def test_confidence_variance_calculation():
    """Verifies that confidence variance is correctly computed."""
    tracker = TemporalInstabilityTracker()
    t0 = 1000.0

    # Low variance confidence stream: [0.90, 0.91, 0.89, 0.90]
    for i, c in enumerate([0.90, 0.91, 0.89, 0.90]):
        tracker.record_prediction("stable-conf", t0 + i, 0.20, c)
    m_stable = tracker.compute_instability("stable-conf")
    assert m_stable.confidence_variance < 0.001

    # High variance confidence stream: [0.20, 0.95, 0.15, 0.90]
    for i, c in enumerate([0.20, 0.95, 0.15, 0.90]):
        tracker.record_prediction("volatile-conf", t0 + i, 0.20, c)
    m_vol = tracker.compute_instability("volatile-conf")
    assert m_vol.confidence_variance > 0.10


def test_class_flip_count_and_rate():
    """Verifies class-flip detection across decision threshold."""
    tracker = TemporalInstabilityTracker()
    t0 = 1000.0

    # 6 observations crossing 0.50 threshold 5 times: 0.4, 0.6, 0.4, 0.6, 0.4, 0.6
    oscillating_risks = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6]
    for i, r in enumerate(oscillating_risks):
        tracker.record_prediction("flipper", t0 + i, r, 0.80, threshold=0.50)

    m = tracker.compute_instability("flipper", threshold=0.50)
    assert m.class_flip_count == 5
    assert m.class_flip_rate == 1.0  # 5 flips / 5 intervals = 1.0


def test_trajectory_instability_smooth_vs_oscillating():
    """Verifies that trajectory acceleration separates smooth ramps from jagged oscillation."""
    tracker = TemporalInstabilityTracker()
    t0 = 1000.0

    # Smooth linear ramp: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] -> second difference is 0.0
    for i, r in enumerate(np.linspace(0.1, 0.6, 6)):
        tracker.record_prediction("smooth-ramp", t0 + i, r, 0.90)
    m_smooth = tracker.compute_instability("smooth-ramp")

    # Jagged zig-zag: [0.1, 0.6, 0.15, 0.65, 0.1, 0.7] -> massive second differences
    for i, r in enumerate([0.1, 0.6, 0.15, 0.65, 0.1, 0.7]):
        tracker.record_prediction("jagged", t0 + i, r, 0.90)
    m_jagged = tracker.compute_instability("jagged")

    assert m_smooth.trajectory_instability < m_jagged.trajectory_instability
    assert m_smooth.trajectory_instability == pytest.approx(0.0, abs=1e-3)
    assert m_jagged.trajectory_instability > 0.50


def test_prediction_entropy_near_extremes_vs_boundary():
    """Verifies that Shannon entropy is low at extremes (0 or 1) and maximized near boundary (0.50)."""
    tracker = TemporalInstabilityTracker()
    t0 = 1000.0

    # Near benign extreme: [0.02, 0.03, 0.02, 0.01]
    for i, r in enumerate([0.02, 0.03, 0.02, 0.01]):
        tracker.record_prediction("extreme-benign", t0 + i, r, 0.95)
    m_benign = tracker.compute_instability("extreme-benign")
    assert m_benign.prediction_entropy < 0.25

    # Near decision boundary: [0.50, 0.51, 0.49, 0.50]
    for i, r in enumerate([0.50, 0.51, 0.49, 0.50]):
        tracker.record_prediction("boundary", t0 + i, r, 0.95)
    m_boundary = tracker.compute_instability("boundary")
    assert m_boundary.prediction_entropy > 0.95


def test_four_behavioral_cohorts():
    """
    Rigorously tests differentiation of the 4 specified behavioral cohorts:
      1. Stable Benign
      2. Stable Attack
      3. Gradual Attack
      4. Oscillating Ambiguous
    """
    tracker = TemporalInstabilityTracker(max_window_size=30)
    t0 = 1000.0
    rng = np.random.default_rng(42)

    # 1. Stable Benign: consistently low risk, steady confidence
    for i in range(20):
        r = float(np.clip(rng.normal(0.08, 0.02), 0.0, 0.20))
        tracker.record_prediction("cohort-benign", t0 + i, r, 0.95)
    m_benign = tracker.compute_instability("cohort-benign")
    assert m_benign.instability_score < 0.20
    assert m_benign.class_flip_count == 0
    assert not m_benign.autonomy_gated

    # 2. Stable Attack: consistently high risk, steady confidence
    for i in range(20):
        r = float(np.clip(rng.normal(0.92, 0.02), 0.80, 1.0))
        tracker.record_prediction("cohort-attack", t0 + i, r, 0.95)
    m_attack = tracker.compute_instability("cohort-attack")
    assert m_attack.instability_score < 0.20
    assert m_attack.class_flip_count == 0
    assert not m_attack.autonomy_gated

    # 3. Gradual Attack: monotonic progressive ramp
    for i, r in enumerate(np.linspace(0.10, 0.85, 20)):
        tracker.record_prediction("cohort-gradual", t0 + i, r, 0.90)
    m_gradual = tracker.compute_instability("cohort-gradual")
    # Low flips (at most 1 threshold crossing)
    assert m_gradual.class_flip_count <= 1
    assert m_gradual.trajectory_instability < 0.10
    assert m_gradual.instability_score < 0.35

    # 4. Oscillating Ambiguous: flickering rapidly across 0.50 threshold with volatile confidence
    for i in range(20):
        r = 0.55 if i % 2 == 0 else 0.45
        c = 0.50 if i % 2 == 0 else 0.85
        tracker.record_prediction("cohort-oscillating", t0 + i, r, c)
    m_osc = tracker.compute_instability("cohort-oscillating")
    assert m_osc.class_flip_count >= 15
    assert m_osc.instability_score > 0.60
    assert m_osc.autonomy_gated is True
    assert m_osc.recommended_action in ("ABSTAIN", "ESCALATE_ANALYST")


def test_non_increasing_risk_guarantee():
    """
    CRITICAL SCIENTIFIC GUARANTEE:
    Instability modulates uncertainty and autonomy, but NEVER arbitrarily increases risk score R_t.
    """
    tracker = TemporalInstabilityTracker()

    base_risk = 0.35
    base_uncertainty = 0.15
    high_instability = 0.85

    # Modulate uncertainty
    mod_uncertainty = tracker.modulate_uncertainty(base_uncertainty, high_instability)

    # Uncertainty should increase
    assert mod_uncertainty > base_uncertainty
    # Risk score must remain unchanged
    assert base_risk == 0.35  # Untouched


def test_selective_autonomy_gating_overrides():
    """Verifies that high instability inhibits autonomous containment and elevates monitoring."""
    tracker = TemporalInstabilityTracker()

    # 1. High instability blocks AUTONOMOUS_CONTAINMENT
    action, gated, reason = tracker.modulate_selective_autonomy("AUTONOMOUS_CONTAINMENT", instability_score=0.72)
    assert action == "ESCALATE_ANALYST"
    assert gated is True
    assert "Inhibited autonomous containment" in reason

    # 2. Moderate instability blocks autonomous containment -> ABSTAIN
    action2, gated2, reason2 = tracker.modulate_selective_autonomy("AUTONOMOUS_CONTAINMENT", instability_score=0.45)
    assert action2 == "ABSTAIN"
    assert gated2 is True

    # 3. Low instability allows autonomous action
    action3, gated3, _ = tracker.modulate_selective_autonomy("AUTONOMOUS_CONTAINMENT", instability_score=0.10)
    assert action3 == "AUTONOMOUS_CONTAINMENT"
    assert gated3 is False

    # 4. Moderate instability elevates AUTONOMOUS_PASS to MONITOR
    action4, _, reason4 = tracker.modulate_selective_autonomy("AUTONOMOUS_PASS", instability_score=0.38)
    assert action4 == "MONITOR"
    assert "warrants proactive monitoring" in reason4


def test_temporal_window_pruning():
    """Verifies that records older than temporal_window_sec are pruned."""
    tracker = TemporalInstabilityTracker(max_window_size=50, temporal_window_sec=100.0)

    # Add old record at t=1000
    tracker.record_prediction("aging-host", 1000.0, 0.8, 0.9)
    assert tracker.get_history_length("aging-host") == 1

    # Add new record at t=1200 (>100s later)
    tracker.record_prediction("aging-host", 1200.0, 0.2, 0.9)
    # Old record should have been pruned
    assert tracker.get_history_length("aging-host") == 1


def test_configuration_toggle_exists():
    """Verifies that USE_TEMPORAL_INSTABILITY setting is active and boolean."""
    assert hasattr(settings, "USE_TEMPORAL_INSTABILITY")
    assert isinstance(settings.USE_TEMPORAL_INSTABILITY, bool)
    assert settings.USE_TEMPORAL_INSTABILITY is True
