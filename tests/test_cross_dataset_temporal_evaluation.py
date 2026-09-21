from __future__ import annotations
"""
Unit and Integration Tests for Phase 7 / RQ2: Cross-Dataset & Temporal Generalization
--------------------------------------------------------------------------------------
Validates:
  1. Partition loading, schema mapping, and provenance checksums.
  2. FlowRecord feature conversion and canonical OCSF representation.
  3. OCSF protocol signatures (Slowloris/GoldenEye exhaustion, payload floods).
  4. Welford online dynamic behavioral drift tracking (ΔD).
  5. Validation threshold tuning without test leakage.
  6. Paired permutation testing and effect size computation.
  7. Hypothesis verification: Baselines degrade > 30%, AHRAS bounds degradation <= 15%.
  8. Artifact JSON schema and field integrity.
"""

import os
import json
import math
import pytest
import numpy as np

from evaluation.cross_dataset_temporal_experiment import (
    CrossDatasetTemporalExperiment,
    FlowRecord,
    PartitionMetrics,
    ModelGeneralizationResult,
)
from detection.statistical_engine.peer_group import _WelfordAccumulator
from evaluation.dataset_loader import compute_file_sha256

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def experiment():
    return CrossDatasetTemporalExperiment(seed=42)


@pytest.fixture(scope="module")
def sample_partitions(experiment):
    train, val, in_test = experiment.load_in_domain_partition(sample_size=1000)
    temp_test = experiment.load_temporal_shift_partition(sample_size=1000)
    cross_test = experiment.load_cross_dataset_partition(sample_size=1000)
    return {
        "train": train,
        "val": val,
        "in_test": in_test,
        "temp_test": temp_test,
        "cross_test": cross_test,
    }


def test_partition_loading_and_shapes(sample_partitions):
    """Verify that all 3 partitions load with non-zero samples and expected class balances."""
    train = sample_partitions["train"]
    val = sample_partitions["val"]
    in_test = sample_partitions["in_test"]
    temp_test = sample_partitions["temp_test"]
    cross_test = sample_partitions["cross_test"]

    assert len(train) > 0
    assert len(val) > 0
    assert len(in_test) > 0
    assert len(temp_test) > 0
    assert len(cross_test) > 0

    # Verify both benign and attack labels exist in train, temp, and cross partitions
    assert any(r.label == 0 for r in train)
    assert any(r.label == 1 for r in train)
    assert any(r.label == 0 for r in temp_test)
    assert any(r.label == 1 for r in temp_test)
    assert any(r.label == 0 for r in cross_test)
    assert any(r.label == 1 for r in cross_test)


def test_flow_record_feature_vector():
    """Verify feature vector serialization format, dimensions, and numerical validity."""
    rec = FlowRecord(
        src_ip="192.168.1.10",
        dst_port=80.0,
        duration_sec=12.5,
        packet_count=15.0,
        bwd_packets=10.0,
        byte_count=1250.0,
        pps=2.0,
        syn_flag=1.0,
        ack_flag=1.0,
        label=1,
        attack_category="DoS slowloris",
        timestamp=1700000000.0,
    )
    vec = rec.to_feature_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    assert vec[0] == 80.0
    assert vec[1] == 12.5


def test_ocsf_signature_rules(experiment):
    """Verify OCSF canonical signature detector flags Slowloris/GoldenEye and payload bursts."""
    # Test 1: Slow HTTP connection exhaustion (port 80, long duration, tiny byte/s)
    slowloris_flow = FlowRecord(
        src_ip="10.0.0.1",
        dst_port=80.0,
        duration_sec=15.0,
        packet_count=4.0,
        bwd_packets=2.0,
        byte_count=200.0,
        pps=0.25,
        syn_flag=1.0,
        ack_flag=0.0,
        label=1,
        attack_category="DoS GoldenEye",
        timestamp=1700000000.0,
    )
    score_slow = experiment._get_ocsf_signature_score(slowloris_flow)
    assert score_slow >= 0.80

    # Test 2: Volumetric flood / DoS Hulk (high pps)
    flood_flow = FlowRecord(
        src_ip="10.0.0.2",
        dst_port=80.0,
        duration_sec=0.1,
        packet_count=50.0,
        bwd_packets=0.0,
        byte_count=40000.0,
        pps=500.0,
        syn_flag=1.0,
        ack_flag=0.0,
        label=1,
        attack_category="DoS Hulk",
        timestamp=1700000001.0,
    )
    score_flood = experiment._get_ocsf_signature_score(flood_flow)
    assert score_flood >= 0.80

    # Test 3: Normal ephemeral flow (should return 0.0)
    normal_flow = FlowRecord(
        src_ip="10.0.0.3",
        dst_port=49152.0,
        duration_sec=0.02,
        packet_count=2.0,
        bwd_packets=2.0,
        byte_count=120.0,
        pps=100.0,
        syn_flag=1.0,
        ack_flag=1.0,
        label=0,
        attack_category="Benign",
        timestamp=1700000002.0,
    )
    score_normal = experiment._get_ocsf_signature_score(normal_flow)
    assert score_normal == 0.0


def test_welford_drift_online_mitigation():
    """Verify Welford accumulator adapts to normal background drift while flagging outliers."""
    acc = _WelfordAccumulator()
    rng = np.random.default_rng(42)

    # In-domain background normal values: N(3.5, 0.5^2)
    for x in rng.normal(3.5, 0.5, size=200):
        acc.update(float(x))

    assert acc.n == 200
    assert abs(acc.mean - 3.5) < 0.1
    assert abs(acc.std - 0.5) < 0.1

    # Shifted background in new environment: N(5.5, 0.5^2)
    # Online adaptation assimilates normal background flows
    for x in rng.normal(5.5, 0.5, size=200):
        acc.update(float(x))

    # Mean successfully adapted towards ~4.5
    assert acc.mean > 4.0

    # Severe outlier (attack burst at 9.0) produces z > 2.0
    z = (9.0 - acc.mean) / acc.std
    assert z > 2.0


def test_threshold_optimization(experiment):
    """Verify threshold tuning strictly optimizes F1 on validation data."""
    rng = np.random.default_rng(42)
    y_val = np.array([0] * 80 + [1] * 20, dtype=np.int32)
    # Scores separated by ground truth
    scores = list(np.concatenate([rng.uniform(0.1, 0.4, size=80), rng.uniform(0.6, 0.9, size=20)]))

    tau, best_f1 = experiment._optimize_threshold(y_val, scores)
    assert 0.40 <= tau <= 0.60
    assert best_f1 >= 0.90


def test_paired_permutation_statistical_test(experiment):
    """Verify paired permutation test calculates unbiased p-values and Cohen's d."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    # Base model has higher error than AHRAS
    base_scores = np.array([0.4] * 50 + [0.4] * 50, dtype=np.float64)
    ahras_scores = list(np.array([0.05] * 50 + [0.95] * 50, dtype=np.float64))

    result = experiment._paired_permutation_test(y_true, base_scores, ahras_scores, n_resamples=1000)
    assert "p_value" in result
    assert "cohens_d" in result
    assert result["p_value"] < 0.05
    assert result["statistically_significant"] is True
    assert result["mean_error_reduction"] > 0


def test_full_cross_dataset_temporal_evaluation_run():
    """Verify complete evaluation runner produces verified hypothesis metrics."""
    exp = CrossDatasetTemporalExperiment(seed=42)
    report = exp.run_evaluation(sample_size=1000)

    assert report["experiment_id"] == "EXP-02"
    assert "provenance" in report
    assert "cicids2017_sha256" in report["provenance"]
    assert "unsw_sha256" in report["provenance"]

    hyp = report["hypothesis_verification"]
    assert hyp["baseline_temporal_degradation_exceeds_30_pct"] is True
    assert hyp["ahras_temporal_degradation_bounded_within_15_pct"] is True
    assert hyp["ahras_cross_dataset_degradation_bounded_within_15_pct"] is True
    assert hyp["all_success_criteria_satisfied"] is True


def test_json_report_artifacts_schema():
    """Verify saved report artifacts conform to strict journal schema."""
    report_file = os.path.join(_ROOT, "evaluation", "results", "CROSS_DATASET_TEMPORAL_REPORT.json")
    final_file = os.path.join(_ROOT, "REAL_DATASET_VALIDATION_FINAL.json")

    assert os.path.exists(report_file), f"Missing {report_file}"
    assert os.path.exists(final_file), f"Missing {final_file}"

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["experiment_id"] == "EXP-02"
    assert len(rep["models_evaluated"]) == 5

    with open(final_file, "r", encoding="utf-8") as f:
        val = json.load(f)
    assert val["status"] == "EVALUATED_AUTHENTIC_REAL_DATA"
    assert val["temporal_shift_generalization"]["relative_f1_drop_pct"] <= 15.0
    assert val["cross_dataset_generalization"]["relative_f1_drop_pct"] <= 15.0
    assert val["hypothesis_verification"]["all_success_criteria_satisfied"] is True
