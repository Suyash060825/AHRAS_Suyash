from __future__ import annotations
"""
AHRAS Scientific Evaluation Runner — Phase 8: Open-Set Unknown Attack Detection
-------------------------------------------------------------------------------
Executes Stage 14 / Phase 8 (EXP-03 / RQ3) evaluation:
  - Trains models strictly on Known Classes (Benign, DoS slowloris, DoS Slowhttptest).
  - Tests against completely unseen Zero-Day Attack families (UNSW Exploits/Backdoors/Fuzzers, DoS Hulk).
  - Compares Closed-Set RF, Closed-Set GB, Isolation Forest, Autoencoder, and AHRAS Latent Metric Reasoner.
  - Verifies Zero-Day Unknown Attack Recall >= 75% and False Unknown Rate (FUR) <= 5%.
  - Emits evaluation/results/OPEN_SET_DETECTION_REPORT.json and updates CLAIMS_MANIFEST_FINAL.json (CLM-04).
"""

import os
import sys
import time
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.open_set_detection_experiment import OpenSetDetectionExperiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "OPEN_SET_DETECTION_REPORT.json")
CLAIMS_FILE = os.path.join(_ROOT, "CLAIMS_MANIFEST_FINAL.json")


def run_evaluation() -> dict:
    t_start = time.perf_counter()
    print("=" * 95)
    print("   AHRAS Phase 8 / RQ3: Open-Set Unknown Zero-Day Attack Detection Evaluation")
    print("=" * 95)

    print("[1/4] Initializing Open-Set partitions and isolating held-out zero-day attack families...")
    experiment = OpenSetDetectionExperiment(seed=42)

    print("[2/4] Executing multi-model open-set discrimination across 6 detector architectures...")
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

    # Update CLAIMS_MANIFEST_FINAL.json for CLM-04
    ahras_res = report["models_evaluated"][-1]
    if os.path.exists(CLAIMS_FILE):
        try:
            with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
                claims = json.load(f)
        except Exception:
            claims = {}
    else:
        claims = {}

    claims["CLM-04"] = {
        "claim": "Explicit OOD unknown attack discrimination on held-out families",
        "metric": "zero_day_recall",
        "status": "SUPPORTED" if ahras_res["zero_day_recall"] >= 0.75 else "FAILED",
        "value": ahras_res["zero_day_recall"],
        "false_unknown_rate": ahras_res["false_unknown_rate"],
        "open_set_auroc": ahras_res["open_set_auroc"],
    }
    with open(CLAIMS_FILE, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2, default=_json_default)
    print(f"      Updated claims manifest: {CLAIMS_FILE} (CLM-04)")

    # 4. Display Results Summary Table
    print("\n" + "=" * 95)
    print("           OPEN-SET ZERO-DAY ATTACK DETECTION MATRIX (EXP-03 / RQ3)")
    print("=" * 95)
    print(f"{'Model Architecture':<34} | {'Known F1':<10} | {'Zero-Day Rec':<13} | {'Benign FUR':<11} | {'AUROC':<8} | {'AUPRC':<8}")
    print("-" * 95)
    for m in report["models_evaluated"]:
        name = m["model_name"]
        k_f1 = m["known_macro_f1"]
        zd_rec = m["zero_day_recall"] * 100.0
        fur = m["false_unknown_rate"] * 100.0
        auroc = m["open_set_auroc"]
        auprc = m["open_set_auprc"]
        print(f"{name:<34} | {k_f1:<10.4f} | {zd_rec:>10.2f}%   | {fur:>8.2f}%   | {auroc:<8.4f} | {auprc:<8.4f}")

    print("-" * 95)
    hyp = report["hypothesis_verification"]
    stat = report["statistical_significance_vs_rf"]

    zd_rec_str = f"{ahras_res['zero_day_recall']*100:.2f}%"
    fur_str = f"{ahras_res['false_unknown_rate']*100:.2f}%"
    kf1_str = f"{ahras_res['known_macro_f1']*100:.2f}%"
    auroc_str = f"{ahras_res['open_set_auroc']*100:.2f}%"

    print(f"Unknown-Family Zero-Day Recall >= 75%:  {'CONFIRMED (' + zd_rec_str + ')' if hyp['zero_day_recall_reaches_75_pct'] else 'FAILED'}")
    print(f"False Unknown Rate (FUR) on Benign <= 5%: {'CONFIRMED (' + fur_str + ')' if hyp['false_unknown_rate_bounded_within_5_pct'] else 'FAILED'}")
    print(f"Known-Class Macro F1 >= 85%:            {'CONFIRMED (' + kf1_str + ')' if hyp['known_class_f1_reaches_85_pct'] else 'FAILED'}")
    print(f"Open-Set AUROC >= 85%:                  {'CONFIRMED (' + auroc_str + ')' if hyp['open_set_auroc_reaches_85_pct'] else 'FAILED'}")
    print(f"Paired Permutation Test p-value vs RF:  {stat['p_value']:.6f} (Gain = +{stat['mean_recall_gain']*100:.2f}%, Cohen's d = {stat['cohens_d']:.4f})")
    
    print("\nPer-Family Zero-Day Recall Breakdown:")
    for fam, rec in ahras_res["per_family_recall"].items():
        print(f"  * {fam:<18}: {rec*100:.2f}% recall")

    print(f"\nExecution Completed in:                 {elapsed:.2f} seconds")
    print("=" * 95)

    # Invariant Verification
    assert hyp["zero_day_recall_reaches_75_pct"], f"Zero-Day recall {ahras_res['zero_day_recall']} < 0.75"
    assert hyp["false_unknown_rate_bounded_within_5_pct"], f"FUR {ahras_res['false_unknown_rate']} > 0.05"
    assert hyp["known_class_f1_reaches_85_pct"], f"Known-class F1 {ahras_res['known_macro_f1']} < 0.85"
    assert hyp["open_set_auroc_reaches_85_pct"], f"Open-set AUROC {ahras_res['open_set_auroc']} < 0.85"
    assert stat["statistically_significant"], "AHRAS superiority must be statistically significant (p < 0.05)"
    print(">> ALL OPEN-SET UNKNOWN ATTACK DETECTION SUCCESS CRITERIA STRICTLY SATISFIED.")

    return report


if __name__ == "__main__":
    run_evaluation()
