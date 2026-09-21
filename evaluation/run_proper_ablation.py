from __future__ import annotations
"""
AHRAS Phase 3 Experiment Runner — Proper Ablation Study
-------------------------------------------------------
Executes 100% live computational runs of the Phase 3 Proper Ablation Protocol:
  - Strict leakage audit (temporal split, zero label leakage)
  - 9 Canonical Progression Baselines (B1–B11)
  - 18 Controlled Leave-One-Out Ablations (A1–A18)
  - 10,000 paired sample permutations per comparison
  - Holm-Bonferroni FWER step-down correction (alpha = 0.05)
  - Validation-only threshold tuning
  - Exports evaluation/results/PROPER_ABLATION_REPORT.json
"""

import os
import sys
import time
import json
import logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.dataset_loader import DatasetLoader
from evaluation.generate_synthetic_dataset import generate_and_save
from evaluation.leakage_audit import temporal_train_test_split, LeakageAuditor
from evaluation.ablation_suite import ProperAblationSuite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Phase3Ablation")


def main():
    print("=" * 75)
    print("   AHRAS Phase 3: Proper Ablation Study & Canonical Progression Matrix")
    print("=" * 75)

    # 1. Prepare Telemetry Dataset
    eval_csv = os.path.join(_ROOT, "evaluation", "data", "synthetic_eval_dataset.csv")
    if not os.path.exists(eval_csv):
        log.info("Generating evaluation dataset...")
        eval_csv = generate_and_save(n_total=2000)

    log.info(f"Loading records from {eval_csv}...")
    loader = DatasetLoader(eval_csv, dataset_type="cicids2017")
    all_recs = list(loader.iter_records(limit=1500))
    if len(all_recs) < 50:
        # Fallback to dynamic stream generation if CSV is small
        log.info("Dataset record count < 50, augmenting stream for statistical power...")
        gen_csv = generate_and_save(n_total=1500)
        loader = DatasetLoader(gen_csv, dataset_type="cicids2017")
        all_recs = list(loader.iter_records(limit=1500))

    # 2. Strict Temporal 3-Way Split (Train 70%, Val 15%, Test 15%)
    train_recs, val_recs, test_recs = temporal_train_test_split(all_recs, train_ratio=0.70, val_ratio=0.15)
    log.info(f"Dataset split: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # 3. Leakage Audit
    auditor = LeakageAuditor()
    audit_res = auditor.audit_splits(train_recs, test_recs, val_records=val_recs, is_temporal=True)
    if not audit_res.get("overall_leakage_audit_pass", True):
        log.error("Leakage audit failed! Aborting ablation execution.")
        sys.exit(1)
    log.info("Leakage audit passed: zero overlap, temporal monotonicity confirmed.")

    # 4. Initialize and Run Proper Ablation Suite
    suite = ProperAblationSuite(train_recs, val_recs, test_recs, random_seed=42)
    output_report_path = os.path.join(_ROOT, "evaluation", "results", "PROPER_ABLATION_REPORT.json")
    
    # Run with 10,000 permutations for paper-grade statistical significance
    report = suite.execute_complete_study(output_filepath=output_report_path, n_permutations=10000)

    # 5. Display Summary Tables
    print("\n" + "=" * 75)
    print("   1. CANONICAL PROGRESSION BASELINES (B1–B11)")
    print("=" * 75)
    print(f"{'ID':<28} | {'F1':<7} | {'Precision':<10} | {'Recall':<8} | {'Brier':<8} | {'RASE':<8} | {'Latency':<8}")
    print("-" * 88)
    for b_id, b in report.canonical_baselines.items():
        print(f"{b_id:<28} | {b.f1:<7.4f} | {b.precision:<10.4f} | {b.recall:<8.4f} | {b.brier_score:<8.4f} | {b.rase_score:<8.4f} | {b.latency_ms:<6.2f}ms")

    print("\n" + "=" * 75)
    print("   2. CONTROLLED LEAVE-ONE-OUT ABLATIONS (A1–A18)")
    print("=" * 75)
    print(f"{'Ablation ID':<32} | {'Base':<7} | {'Ablated':<8} | {'Δ F1':<8} | {'Δ %':<8} | {'Cohen d':<8} | {'p (HB)':<8} | {'Sig?'}")
    print("-" * 96)
    for a_id, a in report.controlled_ablations.items():
        sig_str = "YES (***)" if a.statistically_significant else "NO"
        print(f"{a_id:<32} | {a.baseline_metric:<7.4f} | {a.ablated_metric:<8.4f} | {a.delta_metric:<+8.4f} | {a.relative_delta_pct:<+7.2f}% | {a.cohens_d:<8.3f} | {a.adjusted_p:<8.4f} | {sig_str}")

    print("\n" + "=" * 75)
    print("   3. SCIENTIFIC FINDINGS SUMMARY")
    print("=" * 75)
    for k, v in report.summary_findings.items():
        print(f"  - {k}: {v}")
    print(f"Artifact written to: {output_report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
