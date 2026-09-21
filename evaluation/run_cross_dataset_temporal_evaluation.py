from __future__ import annotations
"""
AHRAS Scientific Evaluation Runner — Phase 7: Cross-Dataset & Temporal Generalization
-------------------------------------------------------------------------------------
Executes Stage 13 / Phase 7 (EXP-02 / RQ2) evaluation:
  - In-Domain Partition: CIC-IDS2017 Early Working Hours (Train/Val/Test).
  - Temporal Shift Partition: CIC-IDS2017 Late Afternoon Working Hours (DoS GoldenEye, shifted timing).
  - Cross-Dataset Partition: UNSW-NB15 Multi-Class Network Flows.
  - Compares Random Forest, Gradient Boosting, Isolation Forest, AHRAS Static, and AHRAS Adaptive.
  - Verifies Out-of-Domain degradation bounding (<= 15% drop for AHRAS vs > 30% drop for baselines).
  - Emits evaluation/results/CROSS_DATASET_TEMPORAL_REPORT.json and updates REAL_DATASET_VALIDATION_FINAL.json.
"""

import os
import sys
import time
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.cross_dataset_temporal_experiment import CrossDatasetTemporalExperiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "CROSS_DATASET_TEMPORAL_REPORT.json")
FINAL_VALIDATION_FILE = os.path.join(_ROOT, "REAL_DATASET_VALIDATION_FINAL.json")


def run_evaluation() -> dict:
    t_start = time.perf_counter()
    print("=" * 85)
    print("   AHRAS Phase 7 / RQ2: Cross-Dataset & Temporal Generalization Evaluation")
    print("=" * 85)

    print("[1/4] Loading real-world benchmark datasets (CIC-IDS2017 & UNSW-NB15)...")
    experiment = CrossDatasetTemporalExperiment(seed=42)

    print("[2/4] Executing multi-partition evaluation across 5 detector architectures...")
    report = experiment.run_evaluation(sample_size=4000)

    elapsed = time.perf_counter() - t_start
    report["execution_time_sec"] = round(elapsed, 2)

    # 3. Save Artifacts
    def _json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        return str(o)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"[3/4] Saved report artifact: {REPORT_FILE}")

    # Update REAL_DATASET_VALIDATION_FINAL.json
    ahras_res = report["models_evaluated"][-1]
    rf_res = report["models_evaluated"][0]
    final_validation = {
        "status": "EVALUATED_AUTHENTIC_REAL_DATA",
        "experiment_id": "EXP-02",
        "research_question": "RQ2: Temporal & Cross-Dataset Generalization",
        "provenance": report["provenance"],
        "in_domain_performance": {
            "model": ahras_res["model_name"],
            "f1_score": ahras_res["in_domain"]["f1"],
            "precision": ahras_res["in_domain"]["precision"],
            "recall": ahras_res["in_domain"]["recall"],
            "fpr": ahras_res["in_domain"]["fpr"],
            "pr_auc": ahras_res["in_domain"]["pr_auc"],
            "roc_auc": ahras_res["in_domain"]["roc_auc"],
            "brier_score": ahras_res["in_domain"]["brier_score"],
        },
        "temporal_shift_generalization": {
            "f1_score": ahras_res["temporal_shift"]["f1"],
            "baseline_rf_f1": rf_res["temporal_shift"]["f1"],
            "degradation_ratio": ahras_res["temporal_degradation_ratio"],
            "relative_f1_drop_pct": ahras_res["temporal_relative_f1_drop_pct"],
            "baseline_rf_drop_pct": rf_res["temporal_relative_f1_drop_pct"],
        },
        "cross_dataset_generalization": {
            "f1_score": ahras_res["cross_dataset"]["f1"],
            "baseline_rf_f1": rf_res["cross_dataset"]["f1"],
            "degradation_ratio": ahras_res["cross_dataset_degradation_ratio"],
            "relative_f1_drop_pct": ahras_res["cross_dataset_relative_f1_drop_pct"],
            "baseline_rf_drop_pct": rf_res["cross_dataset_relative_f1_drop_pct"],
        },
        "statistical_significance": report["statistical_significance"],
        "hypothesis_verification": report["hypothesis_verification"],
    }
    with open(FINAL_VALIDATION_FILE, "w", encoding="utf-8") as f:
        json.dump(final_validation, f, indent=2, default=_json_default)
    print(f"      Updated root artifact: {FINAL_VALIDATION_FILE}")

    # 4. Display Results Summary Table
    print("\n" + "=" * 85)
    print("           CROSS-DATASET & TEMPORAL GENERALIZATION MATRIX (EXP-02 / RQ2)")
    print("=" * 85)
    print(f"{'Model Architecture':<26} | {'In-Domain F1':<12} | {'Temporal F1 (Δ)':<17} | {'Cross-Dataset F1 (Δ)':<20}")
    print("-" * 85)
    for m in report["models_evaluated"]:
        name = m["model_name"]
        f1_in = m["in_domain"]["f1"]
        f1_temp = m["temporal_shift"]["f1"]
        drop_temp = m["temporal_relative_f1_drop_pct"]
        f1_cross = m["cross_dataset"]["f1"]
        drop_cross = m["cross_dataset_relative_f1_drop_pct"]

        temp_str = f"{f1_temp:.4f} (-{drop_temp:.1f}%)"
        cross_str = f"{f1_cross:.4f} (-{drop_cross:.1f}%)"
        print(f"{name:<26} | {f1_in:<12.4f} | {temp_str:<17} | {cross_str:<20}")

    print("-" * 85)
    hyp = report["hypothesis_verification"]
    stat_t = report["statistical_significance"]["temporal_shift_vs_rf"]
    stat_c = report["statistical_significance"]["cross_dataset_vs_rf"]

    print(f"Standard Baseline Degradation > 30%: {'CONFIRMED (RF drop > 30%)' if hyp['baseline_temporal_degradation_exceeds_30_pct'] else 'FAILED'}")
    print(f"AHRAS Temporal Degradation <= 15%:    {'CONFIRMED (' + str(ahras_res['temporal_relative_f1_drop_pct']) + '%)' if hyp['ahras_temporal_degradation_bounded_within_15_pct'] else 'FAILED'}")
    print(f"AHRAS Cross-Dataset Degradation <= 15%: {'CONFIRMED (' + str(ahras_res['cross_dataset_relative_f1_drop_pct']) + '%)' if hyp['ahras_cross_dataset_degradation_bounded_within_15_pct'] else 'FAILED'}")
    print(f"Temporal Permutation Test p-value:    {stat_t['p_value']:.6f} (Cohen's d = {stat_t['cohens_d']:.4f})")
    print(f"Cross-Dataset Permutation p-value:    {stat_c['p_value']:.6f} (Cohen's d = {stat_c['cohens_d']:.4f})")
    print(f"Execution Completed in:               {elapsed:.2f} seconds")
    print("=" * 85)

    # Invariant Verification
    assert hyp["baseline_temporal_degradation_exceeds_30_pct"], "Baseline classifiers must exhibit significant degradation (> 30%)"
    assert hyp["ahras_temporal_degradation_bounded_within_15_pct"], f"AHRAS temporal degradation {ahras_res['temporal_degradation_ratio']} > 0.15"
    assert hyp["ahras_cross_dataset_degradation_bounded_within_15_pct"], f"AHRAS cross-dataset degradation {ahras_res['cross_dataset_degradation_ratio']} > 0.15"
    assert stat_t["statistically_significant"], "Temporal shift superiority must be statistically significant (p < 0.05)"
    print(">> ALL CROSS-DATASET & TEMPORAL GENERALIZATION SUCCESS CRITERIA STRICTLY SATISFIED.")

    return report


if __name__ == "__main__":
    run_evaluation()
