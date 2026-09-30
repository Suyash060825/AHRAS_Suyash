#!/usr/bin/env python3
"""
AHRAS Experiment Runner: EXP-22
Threat-Informed Detection Coverage Benchmark
--------------------------------------------
Evaluates AHRAS detection coverage beyond superficial ATT&CK heatmaps,
quantifying implementation-level coverage (IC), telemetry adequacy (TC),
and 5-tier detection depth (L0 to L4) across all 10 core tactics.

Artifacts Produced:
- evaluation/results/DETECTION_COVERAGE_REPORT.json
- publication/tables/detection_coverage.tex
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

from detection_coverage.implementation_catalog import get_default_catalog
from detection_coverage.coverage_report import CoverageReporter, generate_coverage_artifacts
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EXP-22")


def run_benchmark() -> dict:
    log.info("Starting EXP-22: Threat-Informed Detection Coverage Benchmark...")

    result_json_path = _ROOT / "evaluation" / "results" / "DETECTION_COVERAGE_REPORT.json"
    result_latex_path = _ROOT / "publication" / "tables" / "detection_coverage.tex"

    # 1. Run coverage assessment and artifact generation
    reporter = CoverageReporter()
    report_data = reporter.generate_report_data()

    # 2. Attach provenance manifest
    config_dict = {
        "benchmark": "Threat-Informed Detection Coverage",
        "technique_catalog_tactics": 10,
        "depth_hierarchy": ["L0_Blind", "L1_Telemetry", "L2_Fragile", "L3_Validated", "L4_Resilient"],
        "evasion_robustness_threshold": 0.50,
        "empirical_precision_threshold": 0.60,
    }
    manifest = create_manifest(
        experiment_id="EXP-22",
        dataset_name="MITRE_ATTACK_ENTERPRISE_V14",
        dataset_path=str(_ROOT / "coverage" / "implementation_catalog.py"),
        configuration=config_dict,
        result_path=str(result_json_path),
        random_seed=42,
        dataset_version="14.1",
    )
    report_data["manifest"] = manifest.to_dict()

    # Save outputs
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    latex_table = reporter.generate_latex_table(report_data)
    result_latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table + "\n")

    summary = report_data["summary"]
    log.info("EXP-22 Evaluation Complete.")
    log.info(f"Techniques Evaluated: {summary['total_techniques']}")
    log.info(f"Concrete Implementations: {summary['total_implementations']}")
    log.info(f"Observable Implementations: {summary['observable_implementations']} (TC: {summary['macro_telemetry_coverage']*100:.1f}%)")
    log.info(f"Detected Implementations: {summary['detected_implementations']} (IC: {summary['macro_implementation_coverage']*100:.1f}%)")
    log.info(f"Implementation Depth Distribution: {summary['depth_distribution_implementations']}")
    log.info(f"Technique Depth Distribution: {summary['depth_distribution_techniques']}")
    log.info(f"Weakest Tactics: {', '.join(summary['weakest_tactics'])}")
    log.info(f"Report saved to: {result_json_path}")
    log.info(f"LaTeX table saved to: {result_latex_path}")

    return report_data


if __name__ == "__main__":
    run_benchmark()
