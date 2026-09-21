from __future__ import annotations
"""
AHRAS Phase 4 Runner — Controlled Adaptive Weight Learning & Held-Out Evaluation
---------------------------------------------------------------------------------
Executes 100% live computational runs of the Phase 4 protocol:
  - Compares FIXED WEIGHTS vs ADAPTIVE WEIGHTS on held-out test data
  - Records initial/final weights, LR, update counts, training/test sizes
  - Evaluates Precision, Recall, F1, FPR, Brier Score, ECE, RASE
  - Runs 10,000 paired sample permutations for statistical significance
  - Audits shadow validation drift gating and rollback mechanism
  - Exports evaluation/results/ADAPTIVE_WEIGHT_EVALUATION_REPORT.json
"""

import os
import sys
import json
import logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.dataset_loader import DatasetLoader
from evaluation.generate_synthetic_dataset import generate_and_save
from evaluation.leakage_audit import temporal_train_test_split
from evaluation.adaptive_weight_experiment import AdaptiveWeightExperiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Phase4Runner")


def main():
    print("=" * 80)
    print("   AHRAS Phase 4: Controlled Adaptive Weight Learning & Held-Out Evaluation")
    print("=" * 80)

    # 1. Dataset Loading & Preparation
    eval_csv = os.path.join(_ROOT, "evaluation", "data", "synthetic_eval_dataset.csv")
    if not os.path.exists(eval_csv):
        log.info("Generating evaluation dataset...")
        eval_csv = generate_and_save(n_total=2000)

    loader = DatasetLoader(eval_csv, dataset_type="cicids2017")
    all_recs = list(loader.iter_records(limit=2000))
    if len(all_recs) < 50:
        gen_csv = generate_and_save(n_total=2000)
        loader = DatasetLoader(gen_csv, dataset_type="cicids2017")
        all_recs = list(loader.iter_records(limit=2000))

    # 2. Strict 3-Way Temporal Partitioning (Train 70%, Val 15%, Test 15%)
    train_recs, val_recs, test_recs = temporal_train_test_split(all_recs, train_ratio=0.70, val_ratio=0.15)
    log.info(f"Dataset split: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # 3. Initialize and Execute Phase 4 Experiment
    exp = AdaptiveWeightExperiment(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        learning_rate=0.02,
        random_seed=42,
    )

    output_path = os.path.join(_ROOT, "evaluation", "results", "ADAPTIVE_WEIGHT_EVALUATION_REPORT.json")
    report = exp.run_experiment(output_filepath=output_path, n_permutations=10000)

    # 4. Display Results
    print("\n" + "=" * 80)
    print("   1. EXPERIMENTAL PARAMETERS")
    print("=" * 80)
    print(f"  - Training Size:     {report.training_size} samples")
    print(f"  - Validation Size:   {report.validation_size} samples (Shadow holdout)")
    print(f"  - Test Size:         {report.test_size} samples (Held-out untouched)")
    print(f"  - Learning Rate:     {report.learning_rate}")
    print(f"  - Number of Updates: {report.number_of_updates}")
    print(f"  - Initial Weights:   {report.initial_weights}")
    print(f"  - Final Weights:     {report.final_weights}")

    print("\n" + "=" * 80)
    print("   2. HELD-OUT PERFORMANCE COMPARISON: FIXED vs ADAPTIVE WEIGHTS")
    print("=" * 80)
    fmt = "{:<16} | {:<12} | {:<12} | {:<16}"
    print(fmt.format("Metric", "Fixed Weights", "Adaptive Weights", "Relative Change"))
    print("-" * 62)
    m_fix = report.fixed_weights_metrics
    m_adp = report.adaptive_weights_metrics

    metrics_display = [
        ("Precision", m_fix.precision, m_adp.precision, f"{((m_adp.precision - m_fix.precision)/max(m_fix.precision, 1e-4))*100:+.2f}%"),
        ("Recall", m_fix.recall, m_adp.recall, f"{((m_adp.recall - m_fix.recall)/max(m_fix.recall, 1e-4))*100:+.2f}%"),
        ("F1-Score", m_fix.f1, m_adp.f1, f"{((m_adp.f1 - m_fix.f1)/max(m_fix.f1, 1e-4))*100:+.2f}%"),
        ("FPR", m_fix.fpr, m_adp.fpr, f"{((m_adp.fpr - m_fix.fpr)/max(m_fix.fpr, 1e-4))*100:+.2f}%"),
        ("Brier Score", m_fix.brier_score, m_adp.brier_score, f"{((m_adp.brier_score - m_fix.brier_score)/max(m_fix.brier_score, 1e-4))*100:+.2f}%"),
        ("ECE", m_fix.ece, m_adp.ece, f"{((m_adp.ece - m_fix.ece)/max(m_fix.ece, 1e-4))*100:+.2f}%"),
        ("PR-AUC", m_fix.pr_auc, m_adp.pr_auc, f"{((m_adp.pr_auc - m_fix.pr_auc)/max(m_fix.pr_auc, 1e-4))*100:+.2f}%"),
        ("ROC-AUC", m_fix.roc_auc, m_adp.roc_auc, f"{((m_adp.roc_auc - m_fix.roc_auc)/max(m_fix.roc_auc, 1e-4))*100:+.2f}%"),
        ("RASE Safety", m_fix.rase_score, m_adp.rase_score, f"{((m_adp.rase_score - m_fix.rase_score)/max(m_fix.rase_score, 1e-4))*100:+.2f}%"),
    ]
    for name, f_val, a_val, chg in metrics_display:
        print(fmt.format(name, f"{f_val:.4f}", f"{a_val:.4f}", chg))

    print("\n" + "=" * 80)
    print("   3. STATISTICAL PERMUTATION TEST (10,000 PERMUTATIONS)")
    print("=" * 80)
    st = report.statistical_test
    print(f"  - Observed Statistic (Mean Error Diff): {st.get('observed_statistic', 0.0):.6f}")
    print(f"  - Empirical Two-Sided p-value:          {st.get('raw_p', 1.0):.6f}")
    print(f"  - Standardized Effect Size (Cohen d):   {st.get('effect_size', 0.0):.4f}")
    print(f"  - 95% Bootstrap Confidence Interval:    {st.get('bootstrap_ci', [0, 0])}")

    print("\n" + "=" * 80)
    print("   4. SHADOW VALIDATION SAFETY & DRIFT AUDIT")
    print("=" * 80)
    sa = report.safety_gating_audit
    print(f"  - Drift Freeze Protection Verified:     {'PASSED' if sa.get('drift_detection_triggered') else 'FAILED'}")
    print(f"  - Rollback to Baseline Verified:        {'PASSED' if sa.get('rollback_to_v1_verified') else 'FAILED'}")
    print(f"  - Total Promoted Weight Versions:       {sa.get('version_count')}")

    print(f"\nReport written to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
