#!/usr/bin/env python3
"""
AHRAS Experiment Runner: EXP-24
Semantic Adversarial Mutation & Evasion Robustness Benchmark
-----------------------------------------------------------
Evaluates intrusion detection resilience against semantics-preserving
adversarial perturbations across host process, network traffic, and cloud API vectors.

Artifacts Produced:
- evaluation/results/EVASION_ROBUSTNESS_REPORT.json
- publication/tables/evasion_robustness.tex
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adversarial.evasion_evaluator import EvasionEvaluator, EvasionEvaluationReport
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EXP-24")


def generate_latex_table(report_data: dict) -> str:
    """Formats evasion robustness results into publication LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Adversarial Evasion Robustness (EXP-24): Detection Degradation Across Detection Engines Under Semantics-Preserving Mutations}}",
        r"\label{tab:evasion_robustness}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Detection Engine} & \textbf{Baseline Detection} & \textbf{Mutated Detection} & \textbf{Evasion Rate} & \textbf{Robustness Score} \\",
        r"\midrule",
    ]

    for det in report_data.get("detector_scores", []):
        eng = det["engine_name"].replace("_", r"\_").title()
        b_rate = f"{det['baseline_detection_rate'] * 100:.1f}\\%"
        m_rate = f"{det['mutated_detection_rate'] * 100:.1f}\\%"
        ev_rate = f"{det['evasion_rate'] * 100:.1f}\\%"
        rob = f"{det['robustness_score']:.3f}"
        lines.append(f"{eng} & {b_rate} & {m_rate} & {ev_rate} & \\textbf{{{rob}}} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Top High-Leverage Evasion Mutation Strategies:}} \\",
    ])

    top_strats = report_data.get("strategy_efficacies", [])[:4]
    for st in top_strats:
        s_name = st["strategy_name"].replace("_", r"\_")
        mod = st["modality"].title()
        ev_pct = f"{st['evasion_rate_pct']:.1f}\\%"
        targs = ", ".join(st.get("target_detectors_evaded", [])[:2]) or "None"
        lines.append(f"\\quad \\texttt{{{s_name}}} ({mod}) & \\multicolumn{{2}}{{l}}{{Evasion Success: {ev_pct}}} & \\multicolumn{{2}}{{l}}{{Evaded: \\textit{{{targs}}}}} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def run_benchmark() -> dict:
    log.info("Starting EXP-24: Semantic Adversarial Mutation & Evasion Robustness Benchmark...")

    result_json_path = _ROOT / "evaluation" / "results" / "EVASION_ROBUSTNESS_REPORT.json"
    result_latex_path = _ROOT / "publication" / "tables" / "evasion_robustness.tex"

    evaluator = EvasionEvaluator()
    report: EvasionEvaluationReport = evaluator.run_evaluation()

    report_dict = {
        "experiment_id": "EXP-24",
        "benchmark_name": "Semantic Adversarial Mutation & Evasion Robustness Benchmark",
        "summary": {
            "total_evaluated_vectors": report.total_evaluated_vectors,
            "total_mutation_trials": report.total_mutation_trials,
            "overall_baseline_recall": report.overall_baseline_recall,
            "overall_mutated_recall": report.overall_mutated_recall,
            "overall_evasion_rate": report.overall_evasion_rate,
        },
        "detector_scores": [
            {
                "engine_name": d.engine_name,
                "baseline_detection_rate": d.baseline_detection_rate,
                "mutated_detection_rate": d.mutated_detection_rate,
                "evasion_rate": d.evasion_rate,
                "robustness_score": d.robustness_score,
            }
            for d in report.detector_scores
        ],
        "strategy_efficacies": [
            {
                "strategy_name": s.strategy_name,
                "modality": s.modality,
                "total_trials": s.total_trials,
                "evasions_achieved": s.evasions_achieved,
                "evasion_rate_pct": s.evasion_rate_pct,
                "target_detectors_evaded": s.target_detectors_evaded,
            }
            for s in report.strategy_efficacies
        ],
        "robustness_envelope": [
            {
                "epsilon": e.epsilon,
                "detection_recall_pct": e.detection_recall_pct,
                "f1_score": e.f1_score,
                "evasion_rate_pct": e.evasion_rate_pct,
                "mean_uncertainty": e.mean_uncertainty,
            }
            for e in report.robustness_envelope
        ],
        "actionable_hardening_recommendations": report.hardening_recommendations,
    }

    # Attach Research Manifest
    config_dict = {
        "benchmark": "Adversarial Evasion Robustness",
        "perturbation_budgets": [0.00, 0.05, 0.10, 0.15, 0.20],
        "evaluated_modalities": ["process", "network", "cloud"],
        "semantic_integrity_constraints": "RFC & OS Valid",
    }
    manifest = create_manifest(
        experiment_id="EXP-24",
        dataset_name="AHRAS_SEMANTIC_MUTATION_LAB_V1",
        dataset_path=str(_ROOT / "adversarial" / "mutation_engine.py"),
        configuration=config_dict,
        result_path=str(result_json_path),
        random_seed=42,
        dataset_version="1.0",
    )
    report_dict["manifest"] = manifest.to_dict()

    # Save artifacts
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    latex_table = generate_latex_table(report_dict)
    result_latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table + "\n")

    log.info("EXP-24 Benchmark Complete.")
    log.info(f"Evaluated Vectors: {report.total_evaluated_vectors}")
    log.info(f"Total Mutation Trials: {report.total_mutation_trials}")
    log.info(f"Baseline Recall: {report.overall_baseline_recall*100:.1f}% -> Mutated Recall: {report.overall_mutated_recall*100:.1f}%")
    log.info(f"Overall Evasion Rate: {report.overall_evasion_rate*100:.1f}%")
    for d in report.detector_scores:
        log.info(f"  [{d.engine_name}] Robustness: {d.robustness_score:.3f} (Evasion: {d.evasion_rate*100:.1f}%)")
    log.info(f"Report saved to: {result_json_path}")
    log.info(f"LaTeX table saved to: {result_latex_path}")

    return report_dict


if __name__ == "__main__":
    run_benchmark()
