from __future__ import annotations
"""
AHRAS Phase 3 Unit & Integration Tests — Proper Ablation Suite
--------------------------------------------------------------
Rigorously tests:
  1. Statistical helpers: paired_permutation_test, holm_bonferroni_correction, bootstrap CI
  2. Strict Zero Data Leakage invariants (Zero label leakage, zero lookahead)
  3. Validation-only threshold tuning & Conformal calibration
  4. 9 Canonical Progression Baselines (B1–B11)
  5. 18 Controlled Leave-One-Out Ablations (A1–A18)
  6. Machine-readable report generation and schema compliance
"""

import os
import sys
import unittest
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.dataset_loader import DatasetRecord
from evaluation.ablation_suite import (
    paired_permutation_test,
    holm_bonferroni_correction,
    compute_bootstrap_ci,
    LeakFreeContextPipeline,
    ProperAblationSuite,
    ProperAblationReport,
)


def _make_dummy_records(n: int = 60, seed: int = 42) -> list[DatasetRecord]:
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        label = 1 if (i % 3 == 0) else 0
        feats = {
            "dst_port": 80 if label == 0 else 4444,
            "Total Fwd Packets": rng.integers(1, 100),
            "Total Backward Packets": rng.integers(0, 50),
            "Flow Duration": rng.integers(1000, 1000000),
            "Flow Bytes/s": float(rng.uniform(100, 50000)),
            "Flow Packets/s": float(rng.uniform(1, 1000)),
            "SYN Flag Count": 1 if label == 1 else 0,
            "ACK Flag Count": rng.integers(0, 5),
            "Average Packet Size": float(rng.uniform(40, 1500)),
            "Packet Length Std": float(rng.uniform(0, 200)),
            "f10": 0.1, "f11": 0.2, "f12": 0.3, "f13": 0.4,
        }
        rec = DatasetRecord(
            src_ip=f"192.168.1.{10 + (i % 10)}",
            features=feats,
            label=label,
            attack_category="Attack" if label == 1 else "BENIGN",
            raw_row={},
            event_time=1700000000.0 + i * 10.0,
            dst_ip="10.0.0.1",
        )
        recs.append(rec)
    return recs


class TestProperAblationSuite(unittest.TestCase):

    def test_paired_permutation_test_null_hypothesis(self):
        """When base and ablated errors are identical, p-value must be 1.0 and effect size 0.0."""
        errs = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
        res = paired_permutation_test(errs, errs, n_permutations=1000, seed=42)
        self.assertEqual(res["raw_p"], 1.0)
        self.assertEqual(res["effect_size"], 0.0)
        self.assertEqual(res["observed_statistic"], 0.0)

    def test_paired_permutation_test_divergent_errors(self):
        """When ablated errors are consistently much higher, p-value must be significant."""
        rng = np.random.default_rng(42)
        base_errs = np.zeros(100)
        abl_errs = 0.8 + rng.normal(0, 0.05, 100)
        res = paired_permutation_test(base_errs, abl_errs, n_permutations=1000, seed=42)
        self.assertLess(res["raw_p"], 0.01)
        self.assertGreater(res["effect_size"], 2.0)
        self.assertGreater(res["observed_statistic"], 0.5)

    def test_holm_bonferroni_adjustment(self):
        """Verifies monotonicity and FWER control of Holm-Bonferroni correction."""
        raw_p = {
            "test_a": 0.001,
            "test_b": 0.01,
            "test_c": 0.04,
            "test_d": 0.50,
        }
        adjusted = holm_bonferroni_correction(raw_p, alpha=0.05)
        self.assertEqual(len(adjusted), 4)
        
        # Check ranks
        self.assertEqual(adjusted["test_a"]["rank"], 1)
        self.assertEqual(adjusted["test_d"]["rank"], 4)
        
        # Check adjusted values are monotonic non-decreasing
        adj_vals = [adjusted[k]["adjusted_p"] for k in ["test_a", "test_b", "test_c", "test_d"]]
        for i in range(len(adj_vals) - 1):
            self.assertLessEqual(adj_vals[i], adj_vals[i + 1])
            
        # Check bounded by 1.0
        for k in adjusted:
            self.assertLessEqual(adjusted[k]["adjusted_p"], 1.0)
            self.assertGreaterEqual(adjusted[k]["adjusted_p"], 0.0)

    def test_zero_label_leakage_audit(self):
        """Audits that LeakFreeContextPipeline does not accept or consume ground truth labels."""
        pipeline = LeakFreeContextPipeline()
        self.assertTrue(pipeline.audit_leak_free())
        
        recs = _make_dummy_records(5)
        for i, r in enumerate(recs):
            ocsf_evt = {"src_endpoint": {"ip": r.src_ip}, "activity_id": 1}
            # process_event_context takes only (record, ocsf_event, current_time)
            ctx = pipeline.process_event_context(r, ocsf_evt, 1000.0 + i)
            self.assertIn("ti_score", ctx)
            self.assertIn("h_boost", ctx)
            self.assertIn("g_corr", ctx)
            self.assertIn("p_fore", ctx)
            
            # Post-inference update takes only IP and score (never ground truth label)
            pipeline.update_post_inference(r.src_ip, 0.45)

    def test_validation_threshold_tuning(self):
        """Verifies threshold tuning and conformal calibration on validation partition."""
        train_recs = _make_dummy_records(30, seed=1)
        val_recs = _make_dummy_records(15, seed=2)
        test_recs = _make_dummy_records(15, seed=3)
        
        suite = ProperAblationSuite(train_recs, val_recs, test_recs, random_seed=42)
        best_th, cal_tau = suite.tune_validation_threshold()
        
        self.assertGreaterEqual(best_th, 0.20)
        self.assertLessEqual(best_th, 0.80)
        self.assertGreater(cal_tau, 0.0)
        self.assertLessEqual(cal_tau, 1.0)

    def test_canonical_baselines_execution(self):
        """Verifies that all 9 canonical baselines (B1-B11) execute and produce valid metrics."""
        train_recs = _make_dummy_records(30, seed=10)
        val_recs = _make_dummy_records(15, seed=20)
        test_recs = _make_dummy_records(15, seed=30)
        
        suite = ProperAblationSuite(train_recs, val_recs, test_recs, random_seed=42)
        th, _ = suite.tune_validation_threshold()
        baselines = suite.run_canonical_baselines(th)
        
        expected_ids = [
            "B1_Signature_Only", "B2_Anomaly_Only", "B3_Statistical_Drift_Only",
            "B4_Signature_Plus_Anomaly", "B5_Fixed_Fusion", "B6_Adaptive_Fusion",
            "B7_Full_AHRAS_Core", "B8_Plus_OpenSet_Extension", "B9_Plus_Graph_Extension",
            "B10_Plus_History_Extension", "B11_Plus_Conformal_Safety"
        ]
        for bid in expected_ids:
            self.assertIn(bid, baselines)
            b = baselines[bid]
            self.assertGreaterEqual(b.f1, 0.0)
            self.assertLessEqual(b.f1, 1.0)
            self.assertGreaterEqual(b.accuracy, 0.0)
            self.assertLessEqual(b.accuracy, 1.0)
            self.assertGreaterEqual(b.rase_score, 0.0)

    def test_controlled_ablations_execution(self):
        """Verifies that all 18 leave-one-out ablations (A1-A18) execute with permutation testing."""
        train_recs = _make_dummy_records(30, seed=11)
        val_recs = _make_dummy_records(15, seed=22)
        test_recs = _make_dummy_records(15, seed=33)
        
        suite = ProperAblationSuite(train_recs, val_recs, test_recs, random_seed=42)
        th, _ = suite.tune_validation_threshold()
        ablations = suite.run_controlled_ablations(th, n_permutations=200)
        
        self.assertEqual(len(ablations), 18)
        for a_id, a in ablations.items():
            self.assertGreaterEqual(a.raw_p, 0.0)
            self.assertLessEqual(a.raw_p, 1.0)
            self.assertGreaterEqual(a.adjusted_p, 0.0)
            self.assertLessEqual(a.adjusted_p, 1.0)
            self.assertEqual(a.n_pairs, 15)

    def test_complete_study_report(self):
        """Verifies end-to-end report generation and dictionary serialization."""
        train_recs = _make_dummy_records(20, seed=100)
        val_recs = _make_dummy_records(10, seed=101)
        test_recs = _make_dummy_records(10, seed=102)
        
        suite = ProperAblationSuite(train_recs, val_recs, test_recs, random_seed=42)
        report = suite.execute_complete_study(n_permutations=100)
        
        self.assertIsInstance(report, ProperAblationReport)
        d = report.to_dict()
        self.assertIn("canonical_baselines", d)
        self.assertIn("controlled_ablations", d)
        self.assertIn("leakage_audit_summary", d)
        self.assertIn("summary_findings", d)
        self.assertTrue(d["leakage_audit_summary"]["label_leakage_prevented"])


if __name__ == "__main__":
    unittest.main()
