from __future__ import annotations
"""
AHRAS Scientific Evaluation Runner — Phase 10: Safe Selective Autonomy & Conformal Risk Gating
-----------------------------------------------------------------------------------------------
Executes Stage 16 / Phase 10 (EXP-06 / RQ6) evaluation:
  - Calibrates Split Conformal Selective Risk Gate on holdout validation scenarios (tau* @ 90% coverage).
  - Evaluates 1,000 incident scenarios across 5 comparative response policies.
  - Demonstrates uncalibrated fixed thresholds cause severe false interventions on noisy maintenance spikes.
  - Verifies AHRAS Conformal Risk Gating reduces false interventions by >= 65%, eliminates >= 75%
    of false autonomous containments, and improves RASE safety score by >= 35%.
  - Confirms statistical significance via 10,000 paired sample permutations (p < 0.05).
  - Emits evaluation/results/CONFORMAL_AUTONOMY_REPORT.json and updates CLAIMS_MANIFEST_FINAL.json (CLM-06).
"""

import os
import sys
import time
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.conformal_autonomy_experiment import ConformalAutonomyExperiment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "CONFORMAL_AUTONOMY_REPORT.json")
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
    print("   AHRAS Phase 10 / RQ6: Safe Selective Autonomy & Conformal Risk Gating Evaluation")
    print("=" * 105)

    print("[1/4] Calibrating Split Conformal Selective Risk Gate on 300 holdout scenarios (1 - alpha = 0.90)...")
    experiment = ConformalAutonomyExperiment(seed=42)

    print("[2/4] Executing 1,000 incident scenarios across 5 comparative operational response policies...")
    report = experiment.run_evaluation(n_scenarios=1000, n_calibration=300, target_coverage=0.90)

    elapsed = time.perf_counter() - t_start
    report["execution_time_sec"] = round(elapsed, 2)

    # 3. Save Artifacts
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"[3/4] Saved detailed report artifact: {REPORT_FILE}")

    # Update CLAIMS_MANIFEST_FINAL.json for CLM-06 in root and publication/
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

        claims["CLM-06"] = {
            "claim": clm_entry["claim"],
            "metric": clm_entry["metric"],
            "status": clm_entry["status"],
            "value": clm_entry["value"],
            "false_intervention_reduction_pct": clm_entry["false_intervention_reduction_pct"],
            "rase_improvement_pct": clm_entry["rase_improvement_pct"],
            "attack_containment_rate": clm_entry["attack_containment_rate"],
        }

        with open(c_path, "w", encoding="utf-8") as f:
            json.dump(claims, f, indent=2, default=_json_default)
        print(f"      Updated claims manifest: {c_path} (CLM-06)")

    # 4. Display Results Summary Tables
    print("\n" + "=" * 105)
    print("             SELECTIVE AUTONOMY & CONFORMAL GATING MATRIX (EXP-06 / RQ6)")
    print("=" * 105)
    print(f"{'Response Policy Architecture':<36} | {'Contain%':<9} | {'False Int%':<10} | {'False Auto':<10} | {'Cost Red%':<10} | {'RASE Score':<10} | {'RASE Gain%':<10}")
    print("-" * 105)
    for p in report["policies_evaluated"]:
        name = p["policy_name"].replace("Policy_", "")
        c_rate = f"{p['attack_containment_rate']:.1f}%"
        fi_rate = f"{p['false_intervention_rate']:.1f}%"
        fa_count = f"{p['false_autonomous_containments']}"
        cost_red = f"{p['cost_reduction_pct']:.1f}%"
        rase = f"{p['mean_rase_safety_score']:.4f}"
        gain = f"{p['rase_improvement_pct']:+.1f}%"
        print(f"{name:<36} | {c_rate:<9} | {fi_rate:<10} | {fa_count:<10} | {cost_red:<10} | {rase:<10} | {gain:<10}")
    print("-" * 105)

    hyp = report["hypothesis_verification"]
    stat = report["statistical_significance_vs_uncalibrated"]
    ahras_p = report["policies_evaluated"][-1]
    uncal_p = report["policies_evaluated"][0]

    fa_red = ((uncal_p["false_autonomous_containments"] - ahras_p["false_autonomous_containments"]) / max(1, uncal_p["false_autonomous_containments"])) * 100.0

    print("\nHypothesis & Theoretical Invariant Verification:")
    print(f"  * False Intervention Reduction >= 65%:        {'CONFIRMED (' + str(clm_entry['false_intervention_reduction_pct']) + '%)' if hyp['false_intervention_reduction_reaches_65_pct'] else 'FAILED'}")
    print(f"  * False Autonomous Containment Drop >= 75%:    {'CONFIRMED (' + f'{fa_red:.1f}' + '%)' if hyp['false_autonomous_containment_reduction_reaches_75_pct'] else 'FAILED'}")
    print(f"  * RASE Safety Efficiency Improvement >= 35%:  {'CONFIRMED (' + str(clm_entry['rase_improvement_pct']) + '%)' if hyp['rase_improvement_reaches_35_pct'] else 'FAILED'}")
    print(f"  * Attack Containment Rate >= 90%:             {'CONFIRMED (' + str(clm_entry['attack_containment_rate']) + '%)' if hyp['attack_containment_reaches_90_pct'] else 'FAILED'}")
    print(f"  * Statistical Significance vs Baseline SOAR:  p = {stat['p_value']:.6f} (Mean RASE Gain = +{stat['mean_rase_gain']:.4f}, Cohen's d = {stat['cohens_d']:.4f})")

    print(f"\nExecution Completed in:                         {elapsed:.2f} seconds")
    print("=" * 105)

    # Invariants Verification
    assert hyp["false_intervention_reduction_reaches_65_pct"], "False intervention reduction must reach >= 65%"
    assert hyp["false_autonomous_containment_reduction_reaches_75_pct"], "False autonomous containment reduction must reach >= 75%"
    assert hyp["rase_improvement_reaches_35_pct"], "RASE improvement must reach >= 35%"
    assert hyp["attack_containment_reaches_90_pct"], "Attack containment must reach >= 90%"
    assert stat["statistically_significant"], "AHRAS superiority must be statistically significant (p < 0.05)"
    print(">> ALL SAFE SELECTIVE AUTONOMY & CONFORMAL GATING SUCCESS CRITERIA STRICTLY SATISFIED.")

    return report


if __name__ == "__main__":
    run_evaluation()
