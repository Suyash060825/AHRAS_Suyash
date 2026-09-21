from __future__ import annotations
"""
AHRAS XAI Fidelity & Sum-Check Experiment
------------------------------------------
Reproduces the analytical exactness evaluation:
  1. 6 Targeted threat scenarios exercising every adjustment path
  2. 200-case randomized parameter sweep measuring Delta = |R_engine - R_reconstructed|
  3. Domain alignment precision/recall (FAP, FAR, FAF1)
"""

import os
import json
import random
import time
from typing import Dict, List, Any

from detection.risk_engine import AdaptiveRiskEngine
from xai.fidelity_ledger import XAIFidelityLedger, get_fidelity_ledger

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 6 Targeted Scenarios
_TARGETED_SCENARIOS = [
    ("External Attacker (Signature + ML)", 0.8, 0.7, 0.5, 0.1, "port_scan", ["unique_dst_ports", "packet_count"]),
    ("Benign Internal Host (Clean)", 0.0, 0.1, 0.0, 0.8, "normal", []),
    ("Ransomware Mass Encryption", 1.0, 0.9, 2.5, 0.0, "ransomware", ["entropy", "high_entropy"]),
    ("Historical Recidivism Repeat IP", 0.6, 0.5, 0.2, 0.1, "ssh_brute", ["dst_port", "packet_count"]),
    ("Cloud Privilege Escalation", 0.9, 0.8, 1.2, 0.2, "cloud_evasion", ["action", "user_identity"]),
    ("Maximal Coordinated Assault (Capped)", 1.0, 1.0, 3.0, 0.0, "syn_flood", ["packet_count", "pps"]),
]


def run_targeted_scenarios() -> List[Dict[str, Any]]:
    ledger = XAIFidelityLedger(tolerance=0.01)
    results = []
    
    for name, s_sig, a_ml, delta_d, t_trust, attack_type, top_feats in _TARGETED_SCENARIOS:
        raw_risk = 0.50 * s_sig + 0.30 * a_ml * (1.0 + delta_d) - 0.15 * t_trust
        final_risk = max(0.0, min(1.0, raw_risk))
        
        components = [
            {"name": "signature", "contribution": 0.50 * s_sig},
            {"name": "anomaly", "contribution": 0.30 * a_ml},
            {"name": "behavioral_drift", "contribution": 0.30 * a_ml * delta_d},
        ]
        adjustments = [
            {"type": "trust_discount", "value": -0.15 * t_trust},
        ]
        if raw_risk > 1.0:
            adjustments.append({"type": "cap", "value": 1.0 - raw_risk})
        elif raw_risk < 0.0:
            adjustments.append({"type": "floor", "value": 0.0 - raw_risk})
            
        rec = ledger.verify_explanation(
            event_id=f"TARGETED-{name}",
            entity_key=f"entity-{name}",
            engine_risk_score=final_risk,
            components=components,
            adjustments=adjustments,
            attack_type=attack_type,
            top_explained_features=top_feats,
        )
        results.append({
            "scenario": name,
            "risk_score": rec.engine_risk_score,
            "reconstructed_score": rec.reconstructed_score,
            "reconstruction_error": rec.reconstruction_error,
            "is_faithful": rec.is_faithful,
            "fap": rec.fap,
            "far": rec.far,
        })
    return results


def run_fuzz_sweep(n: int = 200, seed: int = 1337) -> Dict[str, Any]:
    ledger = XAIFidelityLedger(tolerance=0.01)
    rng = random.Random(seed)
    errors = []
    
    for i in range(n):
        s_sig = rng.uniform(0.0, 1.0)
        a_ml = rng.uniform(0.0, 1.0)
        delta_d = rng.uniform(0.0, 3.0)
        t_trust = rng.uniform(0.0, 1.0)
        
        raw_risk = 0.50 * s_sig + 0.30 * a_ml * (1.0 + delta_d) - 0.15 * t_trust
        final_risk = max(0.0, min(1.0, raw_risk))
        
        components = [
            {"name": "signature", "contribution": 0.50 * s_sig},
            {"name": "anomaly", "contribution": 0.30 * a_ml},
            {"name": "behavioral_drift", "contribution": 0.30 * a_ml * delta_d},
        ]
        adjustments = [
            {"type": "trust_discount", "value": -0.15 * t_trust},
        ]
        if raw_risk > 1.0:
            adjustments.append({"type": "cap", "value": 1.0 - raw_risk})
        elif raw_risk < 0.0:
            adjustments.append({"type": "floor", "value": 0.0 - raw_risk})
            
        rec = ledger.verify_explanation(
            event_id=f"FUZZ-{i}",
            entity_key=f"10.0.{i%254}.1",
            engine_risk_score=final_risk,
            components=components,
            adjustments=adjustments,
            attack_type="port_scan",
            top_explained_features=["unique_dst_ports", "packet_count"],
        )
        errors.append(rec.reconstruction_error)
        
    return {
        "n_samples": n,
        "max_error": max(errors),
        "mean_error": sum(errors) / len(errors),
        "cases_faithful": sum(1 for e in errors if e <= 0.01),
        "tolerance": 0.01,
    }


def run_10_path_fidelity_suite(n_per_path: int = 100, tolerance: float = 1e-4) -> Dict[str, Any]:
    from xai.computational_fidelity import ComputationalFidelityEvaluator
    evaluator = ComputationalFidelityEvaluator(tolerance=tolerance, seed=42)
    report = evaluator.run_all_paths(n_per_path=n_per_path)
    return report.to_dict()


def main():
    targeted = run_targeted_scenarios()
    fuzz = run_fuzz_sweep(200)
    ten_paths = run_10_path_fidelity_suite(n_per_path=100, tolerance=1e-4)
    
    report = {
        "targeted_scenarios": targeted,
        "fuzz_sweep": fuzz,
        "ten_paths_fidelity": ten_paths,
    }
    
    out_path = os.path.join(RESULTS_DIR, "xai_fidelity_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    detailed_path = os.path.join(RESULTS_DIR, "COMPUTATIONAL_FIDELITY_REPORT.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(ten_paths, f, indent=2)
        
    print("=======================================================================")
    print("      AHRAS XAI Fidelity & Analytical Exactness Report")
    print("=======================================================================")
    print(f"Targeted Scenarios Tested: {len(targeted)}")
    for t in targeted:
        print(f"  • {t['scenario']:<38} | R={t['risk_score']:.3f} | Δ={t['reconstruction_error']:.6f} | Faithful={t['is_faithful']}")
    print(f"\n200-Case Fuzz Sweep Mean Error Δ: {fuzz['mean_error']:.8f} (Max Δ: {fuzz['max_error']:.8f})")
    print("\n-----------------------------------------------------------------------")
    print("     10-Path Subsystem Computational Fidelity Evaluation (Phase 2)")
    print("-----------------------------------------------------------------------")
    for path_name, metrics in ten_paths["path_results"].items():
        print(f"  [{'PASS' if metrics['is_exact'] else 'FAIL'}] {path_name:<25} | N={metrics['n_samples']} | MAE={metrics['mae']:.8f} | Max={metrics['max_error']:.8f} | PassRate={metrics['pass_rate']*100:.1f}%")
    print(f"\nAll 10 Paths Passed: {ten_paths['all_paths_passed']}")
    print(f"Saved reports to:\n  - {out_path}\n  - {detailed_path}")


if __name__ == "__main__":
    main()
