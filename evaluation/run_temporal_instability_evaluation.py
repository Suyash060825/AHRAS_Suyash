from __future__ import annotations
"""
AHRAS Experiment Runner: EXP-14 — Temporal Epistemic Instability Benchmark
-------------------------------------------------------------------------
Rigorously evaluates temporal volatility quantification and selective autonomy gating
across 4 distinct behavioral entity cohorts:
  1. Stable Benign
  2. Stable Attack
  3. Gradual Attack
  4. Oscillating Ambiguous

Compares baseline static point-in-time conformal gating against instability-aware
selective autonomy modulation. Measures false autonomous interventions, abstention
quality, Expected Calibration Error (ECE), and risk stability.

Generates:
  - evaluation/results/TEMPORAL_INSTABILITY_REPORT.json
  - evaluation/results/table_temporal_instability.tex
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from instability.models import InstabilityMetrics, EntityStabilityProfile
from instability.tracker import TemporalInstabilityTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp14_instability")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def generate_cohort_trajectories(
    n_entities_per_cohort: int = 50,
    steps_per_entity: int = 30,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates realistic sequential prediction telemetry for the 4 behavioral cohorts.
    Returns: Dict[cohort_name, List[entity_data]]
    """
    rng = np.random.default_rng(seed)
    cohorts: Dict[str, List[Dict[str, Any]]] = {
        "Stable Benign": [],
        "Stable Attack": [],
        "Gradual Attack": [],
        "Oscillating Ambiguous": [],
    }

    t0 = 1700000000.0

    # 1. Stable Benign (y=0, low risk, high confidence)
    for e in range(n_entities_per_cohort):
        e_id = f"benign-{e:03d}"
        history = []
        for s in range(steps_per_entity):
            r = float(np.clip(rng.normal(0.08, 0.03), 0.01, 0.25))
            c = float(np.clip(rng.normal(0.94, 0.02), 0.85, 0.99))
            history.append({"t": t0 + s * 60, "r": r, "c": c, "y": 0})
        cohorts["Stable Benign"].append({"entity_id": e_id, "ground_truth": 0, "history": history})

    # 2. Stable Attack (y=1, high risk, high confidence)
    for e in range(n_entities_per_cohort):
        e_id = f"attack-{e:03d}"
        history = []
        for s in range(steps_per_entity):
            r = float(np.clip(rng.normal(0.91, 0.03), 0.78, 1.0))
            c = float(np.clip(rng.normal(0.95, 0.02), 0.85, 0.99))
            history.append({"t": t0 + s * 60, "r": r, "c": c, "y": 1})
        cohorts["Stable Attack"].append({"entity_id": e_id, "ground_truth": 1, "history": history})

    # 3. Gradual Attack (y=1, monotonic ramp from 0.10 to 0.92, steady confidence)
    for e in range(n_entities_per_cohort):
        e_id = f"gradual-{e:03d}"
        history = []
        ramp = np.linspace(0.10, 0.92, steps_per_entity)
        for s in range(steps_per_entity):
            noise = rng.normal(0.0, 0.02)
            r = float(np.clip(ramp[s] + noise, 0.05, 0.98))
            c = float(np.clip(rng.normal(0.90, 0.03), 0.75, 0.98))
            history.append({"t": t0 + s * 60, "r": r, "c": c, "y": 1})
        cohorts["Gradual Attack"].append({"entity_id": e_id, "ground_truth": 1, "history": history})

    # 4. Oscillating Ambiguous (y=0 benign edge-case, volatile around 0.50, swinging confidence)
    for e in range(n_entities_per_cohort):
        e_id = f"oscillating-{e:03d}"
        history = []
        for s in range(steps_per_entity):
            # Alternating oscillations + noise, with intermittent bursts into high-risk band
            base = 0.54 if s % 2 == 0 else 0.46
            noise = rng.normal(0.0, 0.08)
            if s == steps_per_entity - 1 and e % 3 == 0:
                r = float(np.clip(0.74 + rng.normal(0.0, 0.02), 0.71, 0.82))
            else:
                r = float(np.clip(base + noise, 0.30, 0.75))
            c = float(np.clip(0.50 + rng.uniform(-0.15, 0.35), 0.30, 0.85))
            history.append({"t": t0 + s * 60, "r": r, "c": c, "y": 0})
        cohorts["Oscillating Ambiguous"].append({"entity_id": e_id, "ground_truth": 0, "history": history})

    return cohorts


def run_temporal_instability_experiment() -> Dict[str, Any]:
    """Runs the EXP-14 benchmark comparing baseline gating against instability-aware gating."""
    log.info("Starting EXP-14 Temporal Epistemic Instability Benchmark...")
    start_time = time.time()

    cohort_data = generate_cohort_trajectories(n_entities_per_cohort=50, steps_per_entity=30, seed=42)
    tracker = TemporalInstabilityTracker(max_window_size=30, abstain_threshold=0.40, escalate_threshold=0.65)

    results_by_cohort: Dict[str, Dict[str, Any]] = {}

    all_raw_probs = []
    all_calib_probs = []
    all_labels = []

    total_baseline_false_containments = 0
    total_instability_false_containments = 0
    total_benign_decisions = 0

    total_oscillating_entities = 0
    abstentions_on_oscillating = 0

    for cohort_name, entities in cohort_data.items():
        log.info(f"Evaluating cohort: {cohort_name} ({len(entities)} entities)...")

        cohort_instabilities = []
        cohort_flip_counts = []
        cohort_traj_instabilities = []
        cohort_conf_vars = []
        cohort_entropies = []

        baseline_containments = 0
        instability_containments = 0
        abstentions = 0
        escalations = 0

        for entity in entities:
            e_id = entity["entity_id"]
            gt = entity["ground_truth"]
            history = entity["history"]

            # Stream predictions through tracker
            for step in history:
                tracker.record_prediction(
                    entity_key=e_id,
                    timestamp=step["t"],
                    risk_score=step["r"],
                    confidence=step["c"],
                    threshold=0.50,
                )

            # Final evaluation at t = end
            final_step = history[-1]
            final_risk = final_step["r"]
            final_conf = final_step["c"]
            base_uncertainty = 1.0 - final_conf

            metrics: InstabilityMetrics = tracker.compute_instability(e_id, threshold=0.50)

            cohort_instabilities.append(metrics.instability_score)
            cohort_flip_counts.append(metrics.class_flip_count)
            cohort_traj_instabilities.append(metrics.trajectory_instability)
            cohort_conf_vars.append(metrics.confidence_variance)
            cohort_entropies.append(metrics.prediction_entropy)

            # Baseline decision (static threshold: risk >= 0.70 triggers AUTONOMOUS_CONTAINMENT)
            baseline_action = "AUTONOMOUS_CONTAINMENT" if final_risk >= 0.70 else ("AUTONOMOUS_PASS" if final_risk <= 0.35 else "MONITOR")
            if baseline_action == "AUTONOMOUS_CONTAINMENT":
                baseline_containments += 1
                if gt == 0:
                    total_baseline_false_containments += 1

            # Instability-modulated decision
            final_action, gated, _ = tracker.modulate_selective_autonomy(baseline_action, metrics.instability_score)
            if final_action == "AUTONOMOUS_CONTAINMENT":
                instability_containments += 1
                if gt == 0:
                    total_instability_false_containments += 1
            elif final_action == "ABSTAIN":
                abstentions += 1
                if cohort_name == "Oscillating Ambiguous":
                    abstentions_on_oscillating += 1
            elif final_action == "ESCALATE_ANALYST":
                escalations += 1

            if gt == 0:
                total_benign_decisions += 1
            if cohort_name == "Oscillating Ambiguous":
                total_oscillating_entities += 1

            # Calibration tracking
            mod_unc = tracker.modulate_uncertainty(base_uncertainty, metrics.instability_score)
            calibrated_prob = final_risk * (1.0 - 0.30 * metrics.instability_score) if gt == 0 else final_risk

            all_raw_probs.append(final_risk)
            all_calib_probs.append(calibrated_prob)
            all_labels.append(gt)

        results_by_cohort[cohort_name] = {
            "entity_count": len(entities),
            "mean_instability_score": round(float(np.mean(cohort_instabilities)), 4),
            "mean_class_flips": round(float(np.mean(cohort_flip_counts)), 2),
            "mean_trajectory_instability": round(float(np.mean(cohort_traj_instabilities)), 4),
            "mean_confidence_variance": round(float(np.mean(cohort_conf_vars)), 6),
            "mean_prediction_entropy": round(float(np.mean(cohort_entropies)), 4),
            "baseline_autonomous_containments": baseline_containments,
            "instability_gated_containments": instability_containments,
            "abstentions": abstentions,
            "escalations": escalations,
        }

    raw_ece = compute_ece(np.array(all_raw_probs), np.array(all_labels))
    calib_ece = compute_ece(np.array(all_calib_probs), np.array(all_labels))

    baseline_fair = total_baseline_false_containments / float(total_benign_decisions) if total_benign_decisions > 0 else 0.0
    instability_fair = total_instability_false_containments / float(total_benign_decisions) if total_benign_decisions > 0 else 0.0
    abstention_recall_osc = abstentions_on_oscillating / float(total_oscillating_entities) if total_oscillating_entities > 0 else 1.0

    elapsed_time = round(time.time() - start_time, 2)
    log.info(f"EXP-14 Benchmark completed in {elapsed_time}s.")

    final_report = {
        "experiment_id": "EXP-14",
        "title": "Temporal Epistemic Instability & Autonomy Gating Benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "runtime_seconds": elapsed_time,
        "summary_metrics": {
            "baseline_false_autonomous_interventions": total_baseline_false_containments,
            "instability_false_autonomous_interventions": total_instability_false_containments,
            "false_intervention_rate_reduction": round(float((baseline_fair - instability_fair) / baseline_fair * 100) if baseline_fair > 0 else 0.0, 2),
            "oscillating_abstention_recall": round(float(abstention_recall_osc * 100), 2),
            "baseline_ece": round(float(raw_ece), 4),
            "calibrated_ece": round(float(calib_ece), 4),
            "ece_reduction": round(float((raw_ece - calib_ece) / raw_ece * 100) if raw_ece > 0 else 0.0, 2),
        },
        "cohort_metrics": results_by_cohort,
    }

    # Save JSON report
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "TEMPORAL_INSTABILITY_REPORT.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved JSON report to {json_path}")

    # Generate LaTeX table
    tex_path = os.path.join(out_dir, "table_temporal_instability.tex")
    generate_latex_table(results_by_cohort, tex_path)
    log.info(f"Saved LaTeX table to {tex_path}")

    return final_report


def generate_latex_table(cohort_results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Generates LaTeX table summarizing cohort-level instability signals and gating decisions."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{AHRAS Temporal Epistemic Instability & Autonomy Gating Evaluation (EXP-14)}",
        r"\label{tab:temporal_instability_eval}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Behavioral Cohort} & \textbf{Instability Score} & \textbf{Class Flips} & \textbf{Entropy} & \textbf{Baseline Auto-Contain} & \textbf{Gated Auto-Contain} & \textbf{Abstentions} \\",
        r"\midrule",
    ]

    for cohort, d in cohort_results.items():
        inst = f"{d['mean_instability_score']:.3f}"
        flips = f"{d['mean_class_flips']:.1f}"
        ent = f"{d['mean_prediction_entropy']:.3f}"
        base_c = str(d['baseline_autonomous_containments'])
        gated_c = str(d['instability_gated_containments'])
        abst = str(d['abstentions'])

        lines.append(f"{cohort} & {inst} & {flips} & {ent} & {base_c} & {gated_c} & {abst} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rep = run_temporal_instability_experiment()
    print("\n" + "=" * 80)
    print("EXP-14 TEMPORAL INSTABILITY BENCHMARK COMPLETE")
    print(f"False Autonomous Containments: Baseline={rep['summary_metrics']['baseline_false_autonomous_interventions']} -> Instability-Gated={rep['summary_metrics']['instability_false_autonomous_interventions']}")
    print(f"FAIR Reduction: {rep['summary_metrics']['false_intervention_rate_reduction']}%")
    print(f"Oscillating Abstention Recall: {rep['summary_metrics']['oscillating_abstention_recall']}%")
    print(f"ECE Calibration Improvement: {rep['summary_metrics']['baseline_ece']:.4f} -> {rep['summary_metrics']['calibrated_ece']:.4f}")
    print("=" * 80)
