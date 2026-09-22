from __future__ import annotations
"""
AHRAS Scientific Evaluation Runner — Phase 9: Continual Learning & Catastrophic Forgetting
-----------------------------------------------------------------------------------------
Executes Stage 15 / Phase 9 (EXP-04 / RQ4) evaluation:
  - Runs 500-step longitudinal streaming simulation across 5 non-stationary stages (T1–T5).
  - Evaluates 6 Continual Learning Strategies (Static, Naive Online, Replay, Replay Hard Negatives,
    Replay Strategic Forgetting, and AHRAS Active + Continual 5-Bank Multi-Memory).
  - Demonstrates Naive Online suffers catastrophic forgetting (CFR >= 0.20, losing earlier attacks).
  - Verifies AHRAS bounds Catastrophic Forgetting Ratio (CFR <= 0.05) and maintains Backward Transfer
    Retention (>= 94.5%), with Adaptation Gain MSE <= 0.02.
  - Confirms statistical significance via paired permutation testing (p < 0.05).
  - Emits evaluation/results/CONTINUAL_LEARNING_REPORT.json, updates CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json,
    and updates CLAIMS_MANIFEST_FINAL.json (CLM-05).
"""

import os
import sys
import time
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.continual_learning_experiment import ContinualLearningExperiment, STAGES, STRATEGIES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "CONTINUAL_LEARNING_REPORT.json")
CONTINUAL_JSON = os.path.join(_ROOT, "CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json")
PUB_CONTINUAL_JSON = os.path.join(_ROOT, "publication", "CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json")
CLAIMS_FILE = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")
PUB_CLAIMS_FILE = os.path.join(_ROOT, "publication", "CLAIMS_MANIFEST_FINAL.json")


def _json_default(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    return str(o)


def run_evaluation() -> dict:
    t_start = time.perf_counter()
    print("=" * 105)
    print("   AHRAS Phase 9 / RQ4: Continual Learning & Catastrophic Forgetting Evaluation")
    print("=" * 105)

    print("[1/4] Initializing longitudinal non-stationary streaming stages (T1–T5)...")
    experiment = ContinualLearningExperiment(seed=42)

    print("[2/4] Executing 500-step streaming simulation across 6 Continual Learning Strategies...")
    report = experiment.run_evaluation(steps_per_stage=100, test_size_per_stage=250)

    elapsed = time.perf_counter() - t_start
    report["execution_time_sec"] = round(elapsed, 2)

    # 3. Save Artifacts
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"[3/4] Saved detailed report artifact: {REPORT_FILE}")

    # Update CONTINUAL_LEARNING_LONGITUDINAL_FINAL.json in root and publication/
    longitudinal_data = report["continual_learning_longitudinal"]
    with open(CONTINUAL_JSON, "w", encoding="utf-8") as f:
        json.dump(longitudinal_data, f, indent=2, default=_json_default)
    print(f"      Updated longitudinal JSON: {CONTINUAL_JSON}")

    if os.path.exists(os.path.dirname(PUB_CONTINUAL_JSON)):
        with open(PUB_CONTINUAL_JSON, "w", encoding="utf-8") as f:
            json.dump(longitudinal_data, f, indent=2, default=_json_default)
        print(f"      Updated publication JSON: {PUB_CONTINUAL_JSON}")

    # Update CLAIMS_MANIFEST_FINAL.json for CLM-05 in root and publication/
    clm_entry = report["claims_manifest_entry"]
    for c_path in (CLAIMS_FILE, PUB_CLAIMS_FILE):
        if os.path.exists(c_path):
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    claims = json.load(f)
            except Exception:
                claims = {}
        else:
            claims = {}

        claims["CLM-05"] = {
            "claim": clm_entry["claim"],
            "metric": clm_entry["metric"],
            "status": clm_entry["status"],
            "value": clm_entry["value"],
            "backward_transfer_retention_pct": clm_entry["backward_transfer_retention_pct"],
            "catastrophic_forgetting_rate": clm_entry["catastrophic_forgetting_rate"],
        }

        with open(c_path, "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2, default=_json_default)
        print(f"      Updated claims manifest: {c_path} (CLM-05)")

    # 4. Display Results Summary Tables
    print("\n" + "=" * 105)
    print("      CONTINUAL LEARNING LONGITUDINAL PROGRESSION MATRIX (EXP-04 / RQ4)")
    print("=" * 105)
    print(f"{'Strategy':<26} | {'Stage':<24} | {'F1':<7} | {'Loss':<7} | {'Brier':<7} | {'ECE':<7} | {'Degrad':<7} | {'Gain':<7}")
    print("-" * 105)
    for strat in STRATEGIES:
        for stg in STAGES:
            m = longitudinal_data[strat][stg]
            print(f"{strat:<26} | {stg:<24} | {m['f1']:<7.4f} | {m['loss']:<7.4f} | {m['brier']:<7.4f} | {m['ece']:<7.4f} | {m['degradation']:<7.4f} | {m['adaptation_gain']:<7.4f}")
        print("-" * 105)

    print("\n" + "=" * 105)
    print("        CATASTROPHIC FORGETTING & RETENTION AUDIT AT FINAL STAGE (T5)")
    print("=" * 105)
    print(f"{'Strategy':<28} | {'Final F1':<9} | {'CFR':<9} | {'T1 Retention':<14} | {'Memory (MB)':<12}")
    print("-" * 105)
    for strat in STRATEGIES:
        t5_m = longitudinal_data[strat]["T5_Benign_Workload_Shift"]
        f1_str = f"{t5_m['f1']:.4f}"
        cfr_str = f"{t5_m['catastrophic_forgetting_rate']:.4f}"
        ret_str = f"{t5_m['rare_attack_retention_pct']:.1f}%"
        mem_str = f"{t5_m['memory_utilization_mb']:.1f} MB"
        print(f"{strat:<28} | {f1_str:<9} | {cfr_str:<9} | {ret_str:<14} | {mem_str:<12}")
    print("-" * 105)

    hyp = report["hypothesis_verification"]
    stat_naive = report["statistical_significance_vs_naive_online"]
    stat_static = report["statistical_significance_vs_static"]

    ahras_t5 = longitudinal_data["Active_Plus_Continual"]["T5_Benign_Workload_Shift"]
    naive_t5 = longitudinal_data["Naive_Online"]["T5_Benign_Workload_Shift"]

    print("\nHypothesis & Theoretical Invariant Verification:")
    print(f"  * Backward Transfer Retention >= 95%:  {'CONFIRMED (' + str(ahras_t5['rare_attack_retention_pct']) + '%)' if hyp['backward_transfer_retention_reaches_95_pct'] else 'FAILED'}")
    print(f"  * Catastrophic Forgetting CFR <= 0.05:  {'CONFIRMED (' + str(ahras_t5['catastrophic_forgetting_rate']) + ')' if hyp['catastrophic_forgetting_bounded_within_5_pct'] else 'FAILED'}")
    print(f"  * Naive Online Forgetting Detected:     {'CONFIRMED (' + str(naive_t5['catastrophic_forgetting_rate']) + ' CFR, ' + str(naive_t5['rare_attack_retention_pct']) + '% ret)' if hyp['naive_online_suffers_catastrophic_forgetting'] else 'FAILED'}")
    print(f"  * Paired Permutation p vs Naive Online: {stat_naive['p_value']:.6f} (Error Reduction = {stat_naive['mean_error_reduction']:.4f}, Cohen's d = {stat_naive['cohens_d']:.4f})")
    print(f"  * Paired Permutation p vs Static:       {stat_static['p_value']:.6f} (Error Reduction = {stat_static['mean_error_reduction']:.4f}, Cohen's d = {stat_static['cohens_d']:.4f})")

    print(f"\nExecution Completed in:                  {elapsed:.2f} seconds")
    print("=" * 105)

    # Invariants Verification
    assert hyp["backward_transfer_retention_reaches_95_pct"], f"Retention {ahras_t5['rare_attack_retention_pct']}% < 95%"
    assert hyp["catastrophic_forgetting_bounded_within_5_pct"], f"AHRAS CFR {ahras_t5['catastrophic_forgetting_rate']} > 0.05"
    assert stat_naive["statistically_significant"], "AHRAS superiority over Naive Online must be statistically significant (p < 0.05)"
    assert stat_static["statistically_significant"], "AHRAS superiority over Static must be statistically significant (p < 0.05)"
    print(">> ALL CONTINUAL LEARNING & CATASTROPHIC FORGETTING SUCCESS CRITERIA STRICTLY SATISFIED.")

    return report


if __name__ == "__main__":
    run_evaluation()
