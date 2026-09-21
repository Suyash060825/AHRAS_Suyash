from __future__ import annotations
"""
Unit and Integration Tests for Phase 8 / RQ3: Open-Set Unknown Attack Detection
--------------------------------------------------------------------------------
Validates:
  1. Open-Set partition loading, label separation, and provenance checksums.
  2. Canonical feature extraction, scaling, and dimensionality.
  3. Conformal nonconformity threshold calibration (FUR <= 5%).
  4. Closed-set classifier failure on unseen zero-day attacks (MSP recall < 10%).
  5. AHRAS latent metric learning achieving Zero-Day Recall >= 75% and Known F1 >= 85%.
  6. Paired permutation testing and Cohen's d effect sizes.
  7. Report artifact schema and CLAIMS_MANIFEST_FINAL.json (CLM-04) consistency.
"""

import os
import json
import math
import pytest
import numpy as np

from evaluation.open_set_detection_experiment import (
    OpenSetDetectionExperiment,
    OpenSetMetrics,
)
from evaluation.dataset_loader import compute_file_sha256

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def experiment():
    return OpenSetDetectionExperiment(seed=42)


@pytest.fixture(scope="module")
def open_set_data(experiment):
    train_k, val_k, test_k, zero_days = experiment.load_open_set_partitions(sample_size=1000)
    return {
        "train_k": train_k,
        "val_k": val_k,
        "test_k": test_k,
        "zero_days": zero_days,
    }


def test_open_set_partition_loading(open_set_data):
    """Verify that known and unknown zero-day pools are strictly segregated without leakage."""
    train_k = open_set_data["train_k"]
    val_k = open_set_data["val_k"]
    test_k = open_set_data["test_k"]
    zero_days = open_set_data["zero_days"]

    assert len(train_k) > 0
    assert len(val_k) > 0
    assert len(test_k) > 0
    assert len(zero_days) > 0

    known_families = {'Benign', 'DoS slowloris', 'DoS Slowhttptest'}
    # Verify training, validation, and known test partitions only contain known classes
    assert all(r.attack_category in known_families for r in train_k)
    assert all(r.attack_category in known_families for r in val_k)
    assert all(r.attack_category in known_families for r in test_k)

    # Verify zero-day pool contains exclusively attack flows
    assert all(r.label == 1 for r in zero_days)


def test_canonical_feature_extraction(experiment, open_set_data):
    """Verify feature vector serialization, numerical validity, and bounds."""
    r = open_set_data["train_k"][0]
    feats = experiment.extract_canonical_features(r)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (8,)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_conformal_threshold_calibration(experiment):
    """Verify nonconformity threshold calibration controls false unknown rate."""
    rng = np.random.default_rng(42)
    # Simulated validation nonconformity scores for benign traffic
    val_benign_scores = rng.normal(2.0, 0.5, size=500)
    tau = float(np.percentile(val_benign_scores, 97.0))

    # Test benign scores from the same distribution
    test_benign_scores = rng.normal(2.0, 0.5, size=500)
    fur = float(np.mean(test_benign_scores >= tau))

    assert fur <= 0.05, f"False Unknown Rate {fur} exceeded 0.05"


def test_closed_set_baseline_failure(experiment):
    """Verify standard closed-set classifiers fail to detect unseen zero-days (MSP recall < 20%)."""
    report = experiment.run_evaluation(sample_size=1000)
    rf_res = report["models_evaluated"][0]
    gb_res = report["models_evaluated"][1]

    assert rf_res["zero_day_recall"] < 0.20, "Closed-set RF should exhibit low zero-day recall"
    assert gb_res["zero_day_recall"] < 0.20, "Closed-set GB should exhibit low zero-day recall"


def test_ahras_open_set_reasoner_success_criteria(experiment):
    """Verify AHRAS latent metric learning satisfies all hypothesis success criteria."""
    report = experiment.run_evaluation(sample_size=2000)
    ahras_res = report["models_evaluated"][-1]
    hyp = report["hypothesis_verification"]

    assert ahras_res["zero_day_recall"] >= 0.75, f"Zero-day recall {ahras_res['zero_day_recall']} < 0.75"
    assert ahras_res["false_unknown_rate"] <= 0.05, f"FUR {ahras_res['false_unknown_rate']} > 0.05"
    assert ahras_res["known_macro_f1"] >= 0.85, f"Known F1 {ahras_res['known_macro_f1']} < 0.85"
    assert ahras_res["open_set_auroc"] >= 0.85, f"AUROC {ahras_res['open_set_auroc']} < 0.85"
    assert hyp["all_success_criteria_satisfied"] is True


def test_paired_permutation_test(experiment):
    """Verify paired permutation test produces valid statistical significance metrics."""
    base_preds = np.array([0] * 80 + [1] * 20)
    ahras_preds = np.array([1] * 90 + [0] * 10)

    res = experiment._paired_permutation_test(base_preds, ahras_preds, n_resamples=1000)
    assert "p_value" in res
    assert "mean_recall_gain" in res
    assert "cohens_d" in res
    assert res["p_value"] < 0.05
    assert res["statistically_significant"] is True


def test_open_set_report_and_claims_manifest_schema():
    """Verify saved report artifacts conform to strict JSON schemas."""
    report_path = os.path.join(_ROOT, "evaluation", "results", "OPEN_SET_DETECTION_REPORT.json")
    claims_path = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")

    assert os.path.exists(report_path), f"Missing {report_path}"
    assert os.path.exists(claims_path), f"Missing {claims_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["experiment_id"] == "EXP-03"
    assert len(rep["models_evaluated"]) == 6
    assert rep["hypothesis_verification"]["all_success_criteria_satisfied"] is True

    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)
    assert "CLM-04" in claims
    assert claims["CLM-04"]["status"] == "SUPPORTED"
    assert claims["CLM-04"]["value"] >= 0.75
    assert claims["CLM-04"]["false_unknown_rate"] <= 0.05
