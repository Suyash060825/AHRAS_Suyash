#!/usr/bin/env python3
"""
AHRAS Experiment Runner: EXP-23
Telemetry Adequacy & Data Minimality Benchmark
----------------------------------------------
Evaluates:
1. What percentage of telemetry volume can be dropped without losing detection capability?
2. Which fields are indispensable across all 10 ATT&CK tactics?
3. What is the empirical Pareto frontier between telemetry cost and detection coverage?

Artifacts Produced:
- evaluation/results/TELEMETRY_MINIMALITY_REPORT.json
- publication/tables/telemetry_minimality.tex
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

from telemetry.telemetry_requirements import get_default_requirements_registry
from telemetry.feature_importance_mapper import FeatureImportanceMapper
from telemetry.telemetry_analyzer import TelemetryAdequacyAuditor
from telemetry.data_minimality import DataMinimalityOptimizer
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EXP-23")


def generate_latex_table(report_data: dict) -> str:
    """Formats telemetry minimality and Pareto frontier into LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Telemetry Adequacy and Data Minimality (EXP-23): Pareto Frontier Between Ingestion Overhead and Detection Coverage}}",
        r"\label{tab:telemetry_minimality}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Operating Configuration} & \textbf{Fields Retained} & \textbf{Bytes/Event} & \textbf{Coverage Retained} & \textbf{Storage Savings} & \textbf{Pareto Status} \\",
        r"\midrule",
    ]

    for pt in report_data.get("pareto_frontier", []):
        name = pt["operating_point"]
        retained = f"{pt['telemetry_retention_pct']:.1f}\\%"
        b_evt = f"{pt['mean_bytes_per_event']} B"
        cov = f"{pt['detection_coverage_pct']:.1f}\\%"
        savings = f"{pt['storage_savings_pct']:.1f}\\%"
        status = r"\textbf{Optimal}" if pt["is_pareto_optimal"] else "Sub-optimal"
        lines.append(f"{name} & {retained} & {b_evt} & {cov} & {savings} & {status} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{Top Indispensable Telemetry Fields Across All 10 Tactics:}} \\",
    ])

    top_fields = report_data.get("top_indispensable_fields", [])[:5]
    for s in top_fields:
        f_name = s["field_name"].replace("_", r"\_")
        tactics_str = ", ".join(s["tactics_supported"][:3])
        fail_rate = f"{s['failure_rate_if_removed']:.1f}\\%"
        lines.append(f"\\quad \\texttt{{{f_name}}} & \\multicolumn{{2}}{{l}}{{{tactics_str}}} & \\multicolumn{{2}}{{l}}{{Collapse Rate: {fail_rate}}} & \\textit{{{s['classification']}}} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def run_benchmark() -> dict:
    log.info("Starting EXP-23: Telemetry Adequacy & Data Minimality Benchmark...")

    result_json_path = _ROOT / "evaluation" / "results" / "TELEMETRY_MINIMALITY_REPORT.json"
    result_latex_path = _ROOT / "publication" / "tables" / "telemetry_minimality.tex"

    registry = get_default_requirements_registry()
    importance_mapper = FeatureImportanceMapper(registry)
    auditor = TelemetryAdequacyAuditor(registry)
    optimizer = DataMinimalityOptimizer(registry)

    # 1. Feature importance and indispensability
    all_importance = importance_mapper.compute_field_importance()
    top_indispensable = importance_mapper.get_top_indispensable_fields(10)

    # 2. Degraded stress testing
    stress_results = auditor.run_degraded_telemetry_stress_test()

    # 3. Fleet-wide data minimality & Pareto optimization
    minimality_results = optimizer.optimize_fleet_minimality()

    # 4. Compile comprehensive report
    report_data = {
        "experiment_id": "EXP-23",
        "benchmark_name": "Telemetry Adequacy and Data Minimality Benchmark",
        "executive_answers": {
            "volume_reduction_without_detection_loss_pct": minimality_results["overall_volume_reduction_pct"],
            "mean_full_schema_bytes": minimality_results["mean_full_schema_bytes"],
            "mean_minimal_schema_bytes": minimality_results["mean_minimal_schema_bytes"],
            "top_indispensable_fields": [s.field_name for s in top_indispensable],
        },
        "top_indispensable_fields": [
            {
                "field_name": s.field_name,
                "importance_score": s.importance_score,
                "tactics_supported": s.tactics_supported,
                "techniques_affected": s.techniques_affected,
                "classification": s.classification,
                "failure_rate_if_removed": s.failure_rate_if_removed,
            }
            for s in top_indispensable
        ],
        "all_field_importance": [
            {
                "field_name": s.field_name,
                "importance_score": s.importance_score,
                "classification": s.classification,
                "failure_rate_if_removed": s.failure_rate_if_removed,
            }
            for s in all_importance
        ],
        "degraded_telemetry_stress_test": [
            {
                "stage_name": r.stage_name,
                "retained_fields_pct": r.retained_fields_pct,
                "observable_vectors_pct": r.observable_vectors_pct,
                "detection_recall_pct": r.detection_recall_pct,
                "mean_uncertainty": r.mean_uncertainty,
                "f1_score": r.f1_score,
            }
            for r in stress_results
        ],
        "pareto_frontier": [
            {
                "operating_point": p.operating_point,
                "telemetry_retention_pct": p.telemetry_retention_pct,
                "mean_bytes_per_event": p.mean_bytes_per_event,
                "detection_coverage_pct": p.detection_coverage_pct,
                "storage_savings_pct": p.storage_savings_pct,
                "is_pareto_optimal": p.is_pareto_optimal,
            }
            for p in minimality_results["pareto_frontier"]
        ],
        "vector_evaluations": [
            {
                "technique_id": v.technique_id,
                "implementation_id": v.implementation_id,
                "vector_name": v.vector_name,
                "tactic": v.tactic,
                "original_event_bytes": v.original_event_bytes,
                "minimal_event_bytes": v.minimal_event_bytes,
                "volume_reduction_pct": v.volume_reduction_pct,
                "detection_preserved": v.detection_preserved,
            }
            for v in minimality_results["vector_results"]
        ],
    }

    # Attach Research Manifest
    config_dict = {
        "benchmark": "Telemetry Adequacy & Data Minimality",
        "pareto_optimization": True,
        "stress_test_stages": 4,
        "indispensability_threshold": 0.60,
    }
    manifest = create_manifest(
        experiment_id="EXP-23",
        dataset_name="AHRAS_OCSF_TELEMETRY_SCHEMA_V1",
        dataset_path=str(_ROOT / "telemetry" / "telemetry_requirements.py"),
        configuration=config_dict,
        result_path=str(result_json_path),
        random_seed=42,
        dataset_version="1.0",
    )
    report_data["manifest"] = manifest.to_dict()

    # Save artifacts
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    latex_table = generate_latex_table(report_data)
    result_latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table + "\n")

    log.info("EXP-23 Benchmark Complete.")
    log.info(f"Volume Reduction Without Detection Loss: {minimality_results['overall_volume_reduction_pct']}%")
    log.info(f"Mean Full Schema: {minimality_results['mean_full_schema_bytes']} B -> Minimal: {minimality_results['mean_minimal_schema_bytes']} B")
    log.info(f"Indispensable Fields Count: {len(top_indispensable)}")
    log.info(f"Pareto Frontier Points: {len(minimality_results['pareto_frontier'])}")
    log.info(f"Report saved to: {result_json_path}")
    log.info(f"LaTeX table saved to: {result_latex_path}")

    return report_data


if __name__ == "__main__":
    run_benchmark()
