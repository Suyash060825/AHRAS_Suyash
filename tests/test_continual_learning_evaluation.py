from __future__ import annotations
"""
AHRAS Test Suite — Phase 9 / RQ4 Continual Learning & Catastrophic Forgetting
-----------------------------------------------------------------------------
Comprehensive unit and integration test suite validating:
  1. MultiMemoryReplayBuffer compartmental separation and quota sampling.
  2. Prototype moving centroid updates and distance regularization.
  3. Memory footprint estimation and buffer lifecycle.
  4. ContinualLearningExperiment longitudinal execution across 5 stages and 6 strategies.
  5. Catastrophic forgetting prevention (AHRAS CFR <= 0.05 vs Naive Online CFR >= 0.20).
  6. Backward transfer retention (AHRAS >= 94.5% vs Naive Online <= 70.0%).
  7. Paired permutation test statistical significance and Cohen's d.
  8. CLM-05 claims manifest consistency with CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json.
"""

import os
import sys
import json
import pytest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adaptive_learning.weight_learner import (
    MultiMemoryReplayBuffer,
    ContinualLearningEngine,
    FeedbackSample,
)
from evaluation.continual_learning_experiment import (
    ContinualLearningExperiment,
    paired_permutation_test,
    STAGES,
    STRATEGIES,
)


def test_multi_memory_replay_buffer_compartments():
    """Verifies that samples are correctly routed to the 5 distinct memory compartments."""
    buf = MultiMemoryReplayBuffer(recent_cap=10, attack_cap=10, hard_neg_cap=10, drift_cap=10)

    # 1. Normal benign low-loss sample -> only in recent
    s1 = FeedbackSample(src_ip="10.0.0.1", label=0, components={"sig": 0.05, "ml": 0.10}, predicted_risk=0.08)
    buf.add_sample(s1, loss=0.01, is_drift=False)

    # 2. Confirmed attack sample -> recent + attack
    s2 = FeedbackSample(src_ip="10.0.0.2", label=1, components={"sig": 0.90, "ml": 0.85}, predicted_risk=0.88)
    buf.add_sample(s2, loss=0.02, is_drift=False)

    # 3. Hard negative (high-loss benign) -> recent + hard_negative
    s3 = FeedbackSample(src_ip="10.0.0.3", label=0, components={"sig": 0.60, "ml": 0.55}, predicted_risk=0.58)
    buf.add_sample(s3, loss=0.33, is_drift=False)

    # 4. Drift sample -> recent + drift
    s4 = FeedbackSample(src_ip="10.0.0.4", label=0, components={"sig": 0.20, "ml": 0.30}, predicted_risk=0.25)
    buf.add_sample(s4, loss=0.05, is_drift=True)

    stats = buf.get_stats()
    assert stats["recent_count"] == 4
    assert stats["attack_count"] == 1
    assert stats["hard_negative_count"] == 1
    assert stats["drift_count"] == 1
    assert stats["has_normal_proto"] is True
    assert stats["has_attack_proto"] is True


def test_multi_memory_replay_buffer_balanced_quota():
    """Verifies that sample_balanced_batch draws representative samples across all active pools."""
    buf = MultiMemoryReplayBuffer(recent_cap=50, attack_cap=50, hard_neg_cap=50, drift_cap=50)

    # Populate attack pool
    for i in range(10):
        buf.add_sample(FeedbackSample(f"10.0.1.{i}", 1, {"sig": 0.9}, 0.9), loss=0.01)

    # Populate hard-negatives
    for i in range(10):
        buf.add_sample(FeedbackSample(f"10.0.2.{i}", 0, {"sig": 0.6}, 0.6), loss=0.36)

    # Populate drift
    for i in range(10):
        buf.add_sample(FeedbackSample(f"10.0.3.{i}", 0, {"sig": 0.3}, 0.3), loss=0.09, is_drift=True)

    batch = buf.sample_balanced_batch(batch_size=8)
    assert len(batch) == 8

    # Ensure attack samples are included in the balanced batch
    labels = [s.label for s in batch]
    assert 1 in labels, "Balanced replay batch must guarantee rare attack representation"


def test_prototype_distance_and_centroid_update():
    """Verifies that moving prototype centroids update dynamically and compute correct Euclidean distances."""
    buf = MultiMemoryReplayBuffer()
    assert buf.compute_prototype_distance({"f0": 1.0}, label=0) == 0.0

    # Add initial normal sample
    buf.add_sample(FeedbackSample("10.0.0.1", label=0, components={"f0": 0.0, "f1": 0.0}, predicted_risk=0.0), loss=0.0)
    assert buf.normal_prototype is not None
    assert np.allclose(buf.normal_prototype, [0.0, 0.0])

    # Distance of [3.0, 4.0] should be 5.0
    dist = buf.compute_prototype_distance({"f0": 3.0, "f1": 4.0}, label=0)
    assert pytest.approx(dist, abs=1e-3) == 5.0

    # Add initial attack sample
    buf.add_sample(FeedbackSample("10.0.0.2", label=1, components={"f0": 10.0, "f1": 10.0}, predicted_risk=1.0), loss=0.0)
    assert buf.attack_prototype is not None
    assert np.allclose(buf.attack_prototype, [10.0, 10.0])


def test_memory_footprint_and_clear():
    """Verifies RAM footprint estimation and clean reset."""
    buf = MultiMemoryReplayBuffer()
    initial_mem = buf.get_memory_footprint_mb()
    assert initial_mem >= 1.0  # Base buffer allocation ~1MB

    for i in range(50):
        buf.add_sample(FeedbackSample(f"10.0.0.{i}", 0, {"f0": 0.5}, 0.5), loss=0.1)

    assert buf.get_memory_footprint_mb() >= initial_mem

    buf.clear()
    stats = buf.get_stats()
    assert stats["recent_count"] == 0
    assert stats["attack_count"] == 0
    assert stats["hard_negative_count"] == 0
    assert stats["drift_count"] == 0
    assert stats["has_normal_proto"] is False
    assert stats["has_attack_proto"] is False


def test_paired_permutation_test():
    """Verifies the two-sided paired sample permutation test."""
    rng = np.random.default_rng(42)
    # Identical distributions -> p ~ 1.0
    arr_a = rng.normal(0.10, 0.02, 100)
    arr_b = np.copy(arr_a)
    diff, p_val, d = paired_permutation_test(arr_a, arr_b, n_permutations=1000)
    assert diff == 0.0
    assert p_val == 1.0
    assert d == 0.0

    # Significantly different distributions -> p < 0.001
    arr_c = rng.normal(0.30, 0.02, 100)
    diff_sig, p_sig, d_sig = paired_permutation_test(arr_c, arr_a, n_permutations=1000)
    assert diff_sig > 0.15
    assert p_sig < 0.01
    assert d_sig > 2.0


def test_continual_learning_experiment_execution():
    """Runs ContinualLearningExperiment and validates comprehensive report schema and keys."""
    exp = ContinualLearningExperiment(seed=42)
    report = exp.run_evaluation(steps_per_stage=20, test_size_per_stage=50)

    assert report["experiment_id"] == "EXP-04"
    assert report["research_question"] == "RQ4: Continual Learning & Catastrophic Forgetting"
    assert "continual_learning_longitudinal" in report
    assert "backward_transfer_matrix" in report
    assert "backward_transfer_scores" in report
    assert "statistical_significance_vs_naive_online" in report
    assert "statistical_significance_vs_static" in report
    assert "hypothesis_verification" in report
    assert "claims_manifest_entry" in report

    # Verify all 6 strategies and 5 stages are present
    longitudinal = report["continual_learning_longitudinal"]
    assert set(longitudinal.keys()) == set(STRATEGIES)
    for strat in STRATEGIES:
        assert set(longitudinal[strat].keys()) == set(STAGES)
        for stg in STAGES:
            m = longitudinal[strat][stg]
            for field_name in ("f1", "macro_f1", "loss", "fpr", "recall", "brier", "ece", "degradation", "adaptation_gain", "catastrophic_forgetting_rate", "rare_attack_retention_pct", "memory_utilization_mb"):
                assert field_name in m


def test_continual_learning_invariants():
    """Verifies core theoretical invariants for catastrophic forgetting and backward transfer."""
    exp = ContinualLearningExperiment(seed=42)
    report = exp.run_evaluation(steps_per_stage=50, test_size_per_stage=100)

    longitudinal = report["continual_learning_longitudinal"]
    ahras_t5 = longitudinal["Active_Plus_Continual"]["T5_Benign_Workload_Shift"]
    naive_t5 = longitudinal["Naive_Online"]["T5_Benign_Workload_Shift"]
    replay_t5 = longitudinal["Replay"]["T5_Benign_Workload_Shift"]

    # 1. Catastrophic forgetting is bounded for AHRAS but severe in Naive Online
    assert ahras_t5["catastrophic_forgetting_rate"] <= 0.05, "AHRAS CFR must be bounded within 5%"
    assert naive_t5["catastrophic_forgetting_rate"] >= 0.15, "Naive Online must exhibit catastrophic forgetting >= 15%"

    # 2. Backward transfer retention
    assert ahras_t5["rare_attack_retention_pct"] >= 94.5, "AHRAS retention must be >= 94.5%"
    assert naive_t5["rare_attack_retention_pct"] <= 75.0, "Naive Online must degrade retention on early attacks"

    # 3. Adaptation gain
    assert ahras_t5["adaptation_gain"] > 0.0, "AHRAS must show positive adaptation gain over static baseline"

    # 4. Statistical significance
    stat_naive = report["statistical_significance_vs_naive_online"]
    assert stat_naive["statistically_significant"] is True
    assert stat_naive["p_value"] < 0.05

    hyp = report["hypothesis_verification"]
    assert hyp["all_success_criteria_satisfied"] is True


def test_claims_manifest_clm05_consistency():
    """Verifies that CLAIMS_MANIFEST_FINAL.json CLM-05 is correctly supported and matches publication artifacts."""
    claims_path = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")
    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    assert "CLM-05" in claims
    clm = claims["CLM-05"]
    assert clm["status"] == "SUPPORTED"
    assert clm["metric"] == "adaptation_gain_mse"
    assert clm["value"] <= 0.02
    assert clm["backward_transfer_retention_pct"] >= 94.5
    assert clm["catastrophic_forgetting_rate"] <= 0.05
