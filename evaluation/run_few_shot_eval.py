"""
AHRAS Experiment Runner: EXP-30 — Few-Shot Novel Attack Adaptation Benchmark
----------------------------------------------------------------------------
Evaluates rapid zero-day attack adaptation under extreme low-shot regimes:
  - 1-shot, 5-shot, 10-shot, 25-shot

Compares 3 paradigms:
  1. Full Retraining (Monolithic full parameter retraining baseline)
  2. Continual Replay Update (EWC / replay memory blended update)
  3. Prototypical Few-Shot Adaptation (Metric-space prototype adaptation)

Measures:
  - time_to_adapt_ms (computational turnaround time)
  - label_cost (number of supervised exemplar annotations required)
  - new_attack_recall (ability to detect unseen variants of the novel zero-day)
  - old_attack_retention (catastrophic forgetting avoidance on historical attacks)
  - benign_fpr (false positive rate on routine baseline operational traffic)

Outputs:
  - evaluation/results/FEW_SHOT_ADAPTATION_REPORT.json
  - publication/tables/few_shot_adaptation.tex
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_learning.few_shot import (
    FewShotAttackAdapter,
    AdaptationStrategy,
    AdaptationLifecycleStage,
)
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp30_few_shot")


def run_few_shot_benchmark(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    dim = 14

    # Historical distributions
    n_hist_benign = 500
    n_hist_attack = 200
    hist_benign = rng.normal(0.0, 0.08, size=(n_hist_benign, dim))
    hist_attack = rng.normal(0.7, 0.08, size=(n_hist_attack, dim))

    # Novel zero-day attack ground truth distribution (distinct cluster centered at 1.45)
    n_novel_test = 300
    novel_test_ground_truth = rng.normal(1.45, 0.06, size=(n_novel_test, dim))

    shot_regimes = [1, 5, 10, 25]
    paradigms = [
        (AdaptationStrategy.FULL_RETRAINING, "Full Retraining"),
        (AdaptationStrategy.CONTINUAL_REPLAY_UPDATE, "Continual Replay"),
        (AdaptationStrategy.PROTOTYPE_CENTROID, "Prototypical Few-Shot (AHRAS)"),
    ]

    results_table: List[Dict[str, Any]] = []

    for strategy, strat_name in paradigms:
        for shots in shot_regimes:
            log.info(f"Evaluating {strat_name} under {shots}-shot regime...")
            latencies = []
            recalls = []
            retentions = []
            fprs = []

            # 5 cross-validation trials per configuration
            for trial in range(5):
                adapter = FewShotAttackAdapter(embed_dim=dim, temperature=0.1)
                adapter.seed_historical_holdout(hist_benign, hist_attack)

                # Draw k support samples for the novel attack
                support_samples = rng.normal(1.45, 0.06, size=(shots, dim))

                res = adapter.adapt_few_shot(
                    attack_name="ZeroDay-C2-Beacon",
                    support_samples=support_samples,
                    strategy=strategy,
                )

                # Evaluate new attack recall on 300 unseen novel samples
                preds_novel = [adapter.predict_sample(x)[0] == "ZeroDay-C2-Beacon" for x in novel_test_ground_truth]
                novel_recall = float(np.mean(preds_novel))

                # Evaluate old attack retention on historical attacks
                preds_old = [adapter.predict_sample(x)[0] != "benign" for x in hist_attack]
                old_retention = float(np.mean(preds_old))

                # Evaluate benign FPR
                preds_benign = [adapter.predict_sample(x)[0] == "benign" for x in hist_benign]
                benign_fpr = 1.0 - float(np.mean(preds_benign))

                latencies.append(res.adaptation_latency_ms)
                recalls.append(novel_recall)
                retentions.append(old_retention)
                fprs.append(benign_fpr)

            row = {
                "strategy": strat_name,
                "strategy_enum": strategy.value,
                "k_shots": shots,
                "mean_latency_ms": round(float(np.mean(latencies)), 2),
                "std_latency_ms": round(float(np.std(latencies)), 2),
                "new_attack_recall_pct": round(float(np.mean(recalls)) * 100.0, 2),
                "old_attack_retention_pct": round(float(np.mean(retentions)) * 100.0, 2),
                "benign_fpr_pct": round(float(np.mean(fprs)) * 100.0, 2),
            }
            results_table.append(row)

    return {
        "evaluation_summary": results_table,
        "n_hist_benign": n_hist_benign,
        "n_hist_attack": n_hist_attack,
        "n_novel_test": n_novel_test,
    }


def export_latex_table(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Few-Shot Zero-Day Adaptation Benchmark across Training Paradigms (EXP-30)}",
        "\\label{tab:few_shot_adaptation}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "\\textbf{Adaptation Strategy} & \\textbf{Shots ($k$)} & \\textbf{Latency (ms)} & \\textbf{New Recall (\\%)} & \\textbf{Retention (\\%)} & \\textbf{FPR (\\%)} \\\\",
        "\\hline",
    ]

    for r in results:
        strat = r["strategy"]
        k = r["k_shots"]
        lat = f"{r['mean_latency_ms']:.1f} \\pm {r['std_latency_ms']:.1f}"
        rec = f"{r['new_attack_recall_pct']:.1f}"
        ret = f"{r['old_attack_retention_pct']:.1f}"
        fpr = f"{r['benign_fpr_pct']:.1f}"
        tex.append(f"{strat} & {k} & {lat} & {rec} & {ret} & {fpr} \\\\")

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
    log.info("Starting EXP-30: Few-Shot Novel Attack Adaptation Benchmark...")
    bench = run_few_shot_benchmark()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "FEW_SHOT_ADAPTATION_REPORT.json"

    manifest = create_manifest(
        experiment_id="EXP-30",
        dataset_name="Multi-Stage Synthetic Zero-Day Telemetry Stream",
        dataset_path="evaluation/data/dataset_registry.json",
        dataset_version="1.0",
        configuration={"trials_per_config": 5, "embed_dim": 14, "temperature": 0.1},
        result_path=str(report_path),
        random_seed=42,
    )

    final_report = {
        "manifest": manifest.to_dict(),
        "benchmark_results": bench["evaluation_summary"],
        "dataset_metadata": {
            "n_hist_benign": bench["n_hist_benign"],
            "n_hist_attack": bench["n_hist_attack"],
            "n_novel_test": bench["n_novel_test"],
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved machine-readable report to {report_path}")

    latex_path = Path(__file__).resolve().parent.parent / "publication" / "tables" / "few_shot_adaptation.tex"
    export_latex_table(bench["evaluation_summary"], latex_path)
    print("\n=== EXP-30 BENCHMARK SUMMARY ===")
    for row in bench["evaluation_summary"]:
        print(f"[{row['strategy']}] k={row['k_shots']} | Latency: {row['mean_latency_ms']}ms | New Recall: {row['new_attack_recall_pct']}% | Retention: {row['old_attack_retention_pct']}% | FPR: {row['benign_fpr_pct']}%")


if __name__ == "__main__":
    main()
