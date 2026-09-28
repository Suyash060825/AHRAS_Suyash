"""
AHRAS Experiment Runner: EXP-32 — Shadow Model Promotion Safety Benchmark
-------------------------------------------------------------------------
Evaluates deterministic 9-gate safety checks for model promotion:
  - Detection F1 non-degradation
  - False positive rate control
  - Unknown/zero-day OOD recall
  - Expected Calibration Error (ECE)
  - XAI rank stability (Jaccard similarity under perturbation)
  - Operational SLA P95 latency limit (<= 25ms)
  - Memory footprint growth ratio (<= 1.5x)
  - MITRE ATT&CK technique coverage non-decreasing
  - Prequential drift statistic bounds

Compares 4 candidate configurations against active production champion:
  1. Candidate Alpha (Balanced Superior) -> Must PASS & PROMOTE
  2. Candidate Beta (Latency SLA Breached) -> Must REJECT
  3. Candidate Gamma (Uncalibrated Degenerate) -> Must REJECT
  4. Candidate Delta (Coverage Regression) -> Must REJECT

Outputs:
  - evaluation/results/SHADOW_PROMOTION_REPORT.json
  - publication/tables/shadow_promotion.tex
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_learning.shadow_promotion import (
    ModelEvaluationMetrics,
    PromotionPolicy,
    PromotionStatus,
    ShadowModelPromotionPipeline,
)
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp32_shadow_promotion")


def run_shadow_promotion_benchmark() -> Dict[str, Any]:
    champion = ModelEvaluationMetrics(
        model_name="AHRAS-Core-Ensemble",
        version="v1.0.0",
        f1_score=0.965,
        precision=0.970,
        recall=0.960,
        fpr=0.012,
        unknown_ood_recall=0.880,
        ece_calibration=0.075,
        brier_score=0.088,
        xai_stability_jaccard=0.820,
        p50_latency_ms=2.10,
        p95_latency_ms=12.50,
        memory_mb=45.0,
        cpu_cost_factor=1.00,
        mitre_technique_coverage=23,
        drift_statistic=12.4,
    )

    candidates = [
        ModelEvaluationMetrics(
            model_name="AHRAS-Core-Ensemble",
            version="v1.1.0-alpha",
            f1_score=0.974,
            precision=0.978,
            recall=0.970,
            fpr=0.009,
            unknown_ood_recall=0.925,
            ece_calibration=0.058,
            brier_score=0.072,
            xai_stability_jaccard=0.865,
            p50_latency_ms=1.92,
            p95_latency_ms=11.10,
            memory_mb=48.5,
            cpu_cost_factor=1.05,
            mitre_technique_coverage=27,
            drift_statistic=8.2,
        ),
        ModelEvaluationMetrics(
            model_name="AHRAS-Core-Ensemble",
            version="v1.1.0-beta-slow",
            f1_score=0.982,
            precision=0.985,
            recall=0.979,
            fpr=0.007,
            unknown_ood_recall=0.940,
            ece_calibration=0.045,
            brier_score=0.055,
            xai_stability_jaccard=0.880,
            p50_latency_ms=16.50,
            p95_latency_ms=34.80,  # Fails Gate 6 (P95 > 25ms)
            memory_mb=62.0,
            cpu_cost_factor=2.40,
            mitre_technique_coverage=28,
            drift_statistic=10.5,
        ),
        ModelEvaluationMetrics(
            model_name="AHRAS-Core-Ensemble",
            version="v1.1.0-gamma-uncal",
            f1_score=0.971,
            precision=0.975,
            recall=0.967,
            fpr=0.014,
            unknown_ood_recall=0.890,
            ece_calibration=0.235,  # Fails Gate 4 (ECE > 0.15)
            brier_score=0.195,
            xai_stability_jaccard=0.810,
            p50_latency_ms=2.05,
            p95_latency_ms=12.20,
            memory_mb=46.0,
            cpu_cost_factor=1.02,
            mitre_technique_coverage=24,
            drift_statistic=11.0,
        ),
        ModelEvaluationMetrics(
            model_name="AHRAS-Core-Ensemble",
            version="v1.1.0-delta-regress",
            f1_score=0.968,
            precision=0.972,
            recall=0.964,
            fpr=0.010,
            unknown_ood_recall=0.895,
            ece_calibration=0.070,
            brier_score=0.082,
            xai_stability_jaccard=0.825,
            p50_latency_ms=2.00,
            p95_latency_ms=12.00,
            memory_mb=45.5,
            cpu_cost_factor=1.00,
            mitre_technique_coverage=17,  # Fails Gate 8 (Coverage dropped from 23 to 17)
            drift_statistic=11.5,
        ),
    ]

    pipeline = ShadowModelPromotionPipeline(policy=PromotionPolicy())
    pipeline.register_champion(champion)

    eval_results = []
    for cand in candidates:
        pipeline.enroll_shadow_candidate(cand)
        key = f"{cand.model_name}:{cand.version}"
        dec = pipeline.evaluate_promotion(key, champion_override=champion)
        eval_results.append({
            "candidate_version": cand.version,
            "status": dec.status.value,
            "all_gates_passed": dec.all_gates_passed,
            "passed_gate_count": dec.passed_gate_count,
            "total_gate_count": dec.total_gate_count,
            "f1_score": cand.f1_score,
            "p95_latency_ms": cand.p95_latency_ms,
            "ece_calibration": cand.ece_calibration,
            "mitre_coverage": cand.mitre_technique_coverage,
            "rejection_reasons": dec.rejection_reasons,
        })

    return {
        "champion": {
            "name": champion.model_name,
            "version": champion.version,
            "f1_score": champion.f1_score,
            "p95_latency_ms": champion.p95_latency_ms,
            "ece": champion.ece_calibration,
            "mitre_coverage": champion.mitre_technique_coverage,
        },
        "candidates_evaluated": eval_results,
    }


def export_latex_table(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Shadow Model Promotion Invariant Gate Verifications (EXP-32)}",
        "\\label{tab:shadow_promotion}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Candidate Version} & \\textbf{F1} & \\textbf{P95 (ms)} & \\textbf{ECE} & \\textbf{Coverage} & \\textbf{Gates} & \\textbf{Verdict} \\\\",
        "\\hline",
    ]

    for c in results:
        ver = c["candidate_version"]
        f1 = f"{c['f1_score']:.3f}"
        p95 = f"{c['p95_latency_ms']:.1f}"
        ece = f"{c['ece_calibration']:.3f}"
        cov = f"{c['mitre_coverage']}"
        gates = f"{c['passed_gate_count']}/{c['total_gate_count']}"
        verdict = f"\\textbf{{{c['status']}}}"
        tex.append(f"{ver} & {f1} & {p95} & {ece} & {cov} & {gates} & {verdict} \\\\")

    tex.extend([
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    log.info(f"Exported LaTeX table to {output_path}")


def main() -> None:
    log.info("Starting EXP-32: Shadow Model Promotion Safety Benchmark...")
    bench = run_shadow_promotion_benchmark()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "SHADOW_PROMOTION_REPORT.json"

    manifest = create_manifest(
        experiment_id="EXP-32",
        dataset_name="Multi-Metric Shadow Model Benchmark Suite",
        dataset_path="evaluation/data/dataset_registry.json",
        dataset_version="1.0",
        configuration={"gates_count": 9, "candidates_count": 4},
        result_path=str(report_path),
        random_seed=42,
    )

    final_report = {
        "manifest": manifest.to_dict(),
        "champion_baseline": bench["champion"],
        "candidates_evaluated": bench["candidates_evaluated"],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved machine-readable report to {report_path}")

    latex_path = Path(__file__).resolve().parent.parent / "publication" / "tables" / "shadow_promotion.tex"
    export_latex_table(bench["candidates_evaluated"], latex_path)
    print("\n=== EXP-32 BENCHMARK SUMMARY ===")
    for c in bench["candidates_evaluated"]:
        print(f"[{c['candidate_version']}] Verdict: {c['status']} ({c['passed_gate_count']}/{c['total_gate_count']} gates) | Reasons: {c['rejection_reasons']}")


if __name__ == "__main__":
    main()
