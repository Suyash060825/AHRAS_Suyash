"""
Tests for AHRAS Open-Set & Unknown Attack Generalization (Frontier I / EXP-30)
-----------------------------------------------------------------------------
Validates:
  1. OpenMaxEngine MAV fitting, Weibull tail modeling, and unknown class recalibration.
  2. EnergyBasedOODDetector free energy computation, threshold calibration, and OOD flags.
  3. OpenSetClassifier hybrid decisioning between known attack classes and unseen zero-days.
"""

import pytest
import numpy as np

from openset.open_max import OpenMaxEngine, OpenMaxOutput
from openset.energy_detector import EnergyBasedOODDetector, EnergyDetectionOutput
from openset.unknown_classifier import OpenSetClassifier, OpenSetVerdict


class TestOpenMaxEngine:
    def test_fit_and_predict_known_vs_unknown(self):
        rng = np.random.default_rng(42)
        # 3 known classes: 0=Benign, 1=PortScan, 2=BruteForce
        # Generate 60 samples per class with distinct logit peaks
        N = 60
        acts_c0 = rng.normal([8.0, 1.0, 0.5], 0.5, size=(N, 3))
        acts_c1 = rng.normal([1.0, 8.0, 1.0], 0.5, size=(N, 3))
        acts_c2 = rng.normal([0.5, 1.0, 8.0], 0.5, size=(N, 3))

        X = np.vstack([acts_c0, acts_c1, acts_c2])
        y = np.array([0] * N + [1] * N + [2] * N)

        engine = OpenMaxEngine(tail_size=15, alpha_rank=2)
        engine.fit(X, y)
        assert engine.n_classes == 3
        assert len(engine.mavs) == 3

        # Test Known Class 0 instance
        test_known = np.array([8.2, 0.9, 0.4])
        res_known = engine.predict(test_known, unknown_threshold=0.40)
        assert not res_known.is_unknown
        assert res_known.predicted_class == 0

        # Test Distant Outlier / Zero-Day (large distance from all known MAVs)
        test_zero_day = np.array([12.0, 12.0, 12.0])
        res_zd = engine.predict(test_zero_day, unknown_threshold=0.30)
        assert res_zd.is_unknown
        assert res_zd.predicted_class == 3  # Class K is UNKNOWN


class TestEnergyBasedOODDetector:
    def test_free_energy_computation(self):
        detector = EnergyBasedOODDetector(temperature=1.0)
        # Confident sample: large logit peak -> low energy
        high_conf = np.array([10.0, 0.0, 0.0])
        e_conf = detector.compute_energy(high_conf)

        # Ambiguous/flat sample: small logits -> high energy
        flat = np.array([0.0, 0.0, 0.0])
        e_flat = detector.compute_energy(flat)

        assert e_conf < e_flat

    def test_calibrate_and_evaluate(self):
        detector = EnergyBasedOODDetector(temperature=1.0, percentile_threshold=90.0)
        # Known training logits
        known_logits = np.random.normal(size=(100, 4)) + 5.0
        thresh = detector.calibrate(known_logits)
        assert detector.is_calibrated

        # In-distribution sample
        in_sample = np.mean(known_logits, axis=0)
        out_in = detector.evaluate(in_sample)
        assert not out_in.is_out_of_distribution

        # Severe outlier sample (far negative logits)
        outlier = np.array([-15.0, -15.0, -15.0, -15.0])
        out_ood = detector.evaluate(outlier)
        assert out_ood.is_out_of_distribution
        assert out_ood.free_energy > thresh


class TestOpenSetClassifier:
    def test_end_to_end_openset_pipeline(self):
        rng = np.random.default_rng(123)
        # 2 known classes: 0=Benign, 1=SignatureAttack
        N = 50
        logits_0 = rng.normal([6.0, 1.0], 0.3, size=(N, 2))
        logits_1 = rng.normal([1.0, 6.0], 0.3, size=(N, 2))
        X = np.vstack([logits_0, logits_1])
        y = np.array([0] * N + [1] * N)

        clf = OpenSetClassifier(
            class_names=["BENIGN", "SIGNATURE_ATTACK"],
            unknown_prob_threshold=0.40,
        )
        clf.fit(X, y)

        # Predict Known Benign
        v_benign = clf.predict_single(np.array([5.9, 0.9]))
        assert v_benign.is_known
        assert not v_benign.is_zero_day
        assert v_benign.predicted_label == "BENIGN"

        # Predict Known Attack
        v_attack = clf.predict_single(np.array([1.1, 5.8]))
        assert v_attack.is_known
        assert not v_attack.is_zero_day
        assert v_attack.predicted_label == "SIGNATURE_ATTACK"

        # Predict Novel Zero-Day Attack
        v_zeroday = clf.predict_single(np.array([0.0, 0.0]))
        assert v_zeroday.is_zero_day
        assert not v_zeroday.is_known
        assert v_zeroday.predicted_label == "UNKNOWN_ZERO_DAY"
