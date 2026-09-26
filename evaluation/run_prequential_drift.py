#!/usr/bin/env python3
"""
AHRAS Experiment Runner: EXP-25
Streaming Prequential Evaluation & Concept Drift Benchmark
----------------------------------------------------------
Evaluates test-then-train streaming performance without lookahead leakage:
1. Static Model vs Streaming-Updated Model
2. Sudden Drift vs Gradual Drift vs Recurring Seasonality
3. Performance under Verification Label Delays (Immediate, 1-Hour, 1-Day, 1-Week)
4. Memory footprint over time

Artifacts Produced:
- evaluation/results/PREQUENTIAL_DRIFT_REPORT.json
- publication/tables/prequential_drift.tex
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

from streaming.temporal_split import TemporalStreamGenerator, DriftRegimeType
from streaming.prequential_runner import (
    PrequentialEvaluationRunner,
    PrequentialSimulationResult,
    ModelAdaptationStrategy,
)
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("EXP-25")


def generate_latex_table(report_data: dict) -> str:
    """Formats prequential evaluation results into a publication LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Streaming Prequential Evaluation (EXP-25): Model Adaptation Under Concept Drift and Label Latency}}",
        r"\label{tab:prequential_drift}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Adaptation Strategy} & \textbf{Label Delay ($\Delta$)} & \textbf{Macro F1} & \textbf{Final Recall} & \textbf{Brier Score} & \textbf{Adaptation Lag} & \textbf{Memory (MB)} \\",
        r"\midrule",
    ]

    for row in report_data.get("comparisons", []):
        strat = row["strategy"].replace("_", r"\_").title()
        delay = row["delay_label"]
        f1 = f"{row['mean_macro_f1'] * 100:.1f}\\%"
        rec = f"{row['final_recall'] * 100:.1f}\\%"
        brier = f"{row['final_brier_score']:.3f}"
        lag = f"{row['adaptation_delay_steps']} steps" if row["adaptation_delay_steps"] > 0 else "N/A"
        mem = f"{row['mean_memory_mb']:.3f} MB"
        lines.append(f"{strat} & {delay} & {f1} & {rec} & {brier} & {lag} & {mem} \\\\")

    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Performance Breakdown Across Non-Stationary Drift Regimes:}} \\",
    ])

    for reg_row in report_data.get("regime_comparisons", []):
        r_name = reg_row["regime"].replace("_", r"\_").title()
        static_f1 = f"{reg_row['static_f1'] * 100:.1f}\\%"
        stream_f1 = f"{reg_row['streaming_f1'] * 100:.1f}\\%"
        gain = f"+{(reg_row['streaming_f1'] - reg_row['static_f1']) * 100:.1f}\\%"
        lines.append(f"\\quad {r_name} & \\multicolumn{{2}}{{l}}{{Static F1: {static_f1}}} & \\multicolumn{{2}}{{l}}{{Streaming F1: {stream_f1}}} & \\multicolumn{{2}}{{l}}{{Adaptation Gain: \\textbf{{{gain}}}}} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def run_benchmark() -> dict:
    log.info("Starting EXP-25: Streaming Prequential Evaluation & Concept Drift Benchmark...")

    result_json_path = _ROOT / "evaluation" / "results" / "PREQUENTIAL_DRIFT_REPORT.json"
    result_latex_path = _ROOT / "publication" / "tables" / "prequential_drift.tex"

    generator = TemporalStreamGenerator(seed=42)
    # Generate 1000-step canonical non-stationary stream
    full_stream = generator.generate_stream(n_steps=1000)

    # 1. Compare Label Delay Regimes: Immediate (0), 1-Hour (60), 1-Day (240), 1-Week (1000)
    delay_scenarios = [
        ("Immediate Oracle", 0),
        ("1-Hour Delay", 60),
        ("1-Day Delay", 240),
        ("1-Week Delay", 1000),
    ]

    comparisons = []

    # Static baseline (frozen)
    static_runner = PrequentialEvaluationRunner(strategy=ModelAdaptationStrategy.STATIC, label_delay=0)
    static_res = static_runner.run_simulation(full_stream)
    comparisons.append({
        "strategy": "static",
        "delay_label": "No Feedback (Frozen)",
        "label_delay": 0,
        "mean_macro_f1": static_res.mean_macro_f1,
        "final_f1": static_res.final_f1,
        "final_recall": static_res.final_recall,
        "final_brier_score": static_res.final_brier_score,
        "adaptation_delay_steps": 0,
        "mean_memory_mb": static_res.mean_memory_mb,
    })

    # Streaming-updated across delays
    for delay_label, delay_steps in delay_scenarios:
        runner = PrequentialEvaluationRunner(
            strategy=ModelAdaptationStrategy.STREAMING_UPDATED,
            label_delay=delay_steps,
        )
        res = runner.run_simulation(full_stream)
        comparisons.append({
            "strategy": "streaming_updated",
            "delay_label": delay_label,
            "label_delay": delay_steps,
            "mean_macro_f1": res.mean_macro_f1,
            "final_f1": res.final_f1,
            "final_recall": res.final_recall,
            "final_brier_score": res.final_brier_score,
            "adaptation_delay_steps": res.adaptation_delay_steps,
            "mean_memory_mb": res.mean_memory_mb,
        })

    # 2. Compare Regimes Separately: Sudden vs Gradual vs Recurring
    regimes_to_test = [
        (DriftRegimeType.SUDDEN_DRIFT, "Sudden (Abrupt) Drift"),
        (DriftRegimeType.GRADUAL_DRIFT, "Gradual Covariate Drift"),
        (DriftRegimeType.RECURRING_DRIFT, "Recurring Seasonality Drift"),
    ]

    regime_comparisons = []
    for r_type, r_name in regimes_to_test:
        r_stream = generator.generate_stream(
            n_steps=400,
            regimes=[(DriftRegimeType.BASELINE_STATIONARY, 100), (r_type, 300)],
        )
        s_res = PrequentialEvaluationRunner(ModelAdaptationStrategy.STATIC).run_simulation(r_stream)
        u_res = PrequentialEvaluationRunner(ModelAdaptationStrategy.STREAMING_UPDATED, label_delay=10).run_simulation(r_stream)

        regime_comparisons.append({
            "regime": r_type.value,
            "regime_name": r_name,
            "static_f1": s_res.final_f1,
            "streaming_f1": u_res.final_f1,
            "adaptation_gain_f1": round(u_res.final_f1 - s_res.final_f1, 4),
            "adaptation_delay_steps": u_res.adaptation_delay_steps,
            "drift_alerts": u_res.drift_alerts_count,
        })

    # 3. Assemble full report
    report_data = {
        "experiment_id": "EXP-25",
        "benchmark_name": "Streaming Prequential Evaluation & Concept Drift Benchmark",
        "summary": {
            "total_stream_events": 1000,
            "static_mean_f1": static_res.mean_macro_f1,
            "streaming_immediate_mean_f1": comparisons[1]["mean_macro_f1"],
            "adaptation_gain_macro_f1": round(comparisons[1]["mean_macro_f1"] - static_res.mean_macro_f1, 4),
            "peak_memory_mb": static_res.peak_memory_mb,
        },
        "comparisons": comparisons,
        "regime_comparisons": regime_comparisons,
        "time_series_snapshot": static_res.time_series_metrics,
    }

    # Attach Research Manifest
    config_dict = {
        "benchmark": "Streaming Prequential Drift",
        "stream_length": 1000,
        "window_size": 100,
        "delay_scenarios": [0, 60, 240, 1000],
        "drift_regimes": ["sudden", "gradual", "recurring"],
    }
    manifest = create_manifest(
        experiment_id="EXP-25",
        dataset_name="AHRAS_PREQUENTIAL_STREAM_V1",
        dataset_path=str(_ROOT / "streaming" / "temporal_split.py"),
        configuration=config_dict,
        result_path=str(result_json_path),
        random_seed=42,
        dataset_version="1.0",
    )
    report_data["manifest"] = manifest.to_dict()

    # Save outputs
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    latex_table = generate_latex_table(report_data)
    result_latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table + "\n")

    log.info("EXP-25 Benchmark Complete.")
    log.info(f"Static Model F1: {static_res.mean_macro_f1*100:.1f}% -> Streaming Model F1: {comparisons[1]['mean_macro_f1']*100:.1f}%")
    log.info(f"Adaptation Gain: +{(comparisons[1]['mean_macro_f1'] - static_res.mean_macro_f1)*100:.1f}%")
    log.info(f"Report saved to: {result_json_path}")
    log.info(f"LaTeX table saved to: {result_latex_path}")

    return report_data


if __name__ == "__main__":
    run_benchmark()
