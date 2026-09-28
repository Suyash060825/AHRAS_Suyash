"""
AHRAS Experiment Runner: EXP-31 — Adaptive Sensor Acquisition Benchmark
------------------------------------------------------------------------
Evaluates dynamic Value-of-Information (VOI) sensor acquisition against:
  1. Static Network-Only (Minimal baseline: fast, low cost, but high blind spots)
  2. Static Full-Stack Ingestion (Monolithic: collects all modalities always, high cost/latency)
  3. AHRAS Adaptive VOI Acquisition (Dynamically balances Security Gain vs Cost)

Evaluated across 4 operational regimes:
  - Routine Benign (p_threat = 0.05, uncertainty = 0.10, CPU = 20%)
  - Ambiguous Anomaly (p_threat = 0.55, uncertainty = 0.75, CPU = 40%)
  - Critical Server Under Attack (p_threat = 0.90, uncertainty = 0.50, Criticality = 2.0, CPU = 60%)
  - Congestion Flash Crowd (p_threat = 0.70, uncertainty = 0.80, CPU = 95%)

Outputs:
  - evaluation/results/SENSOR_ACQUISITION_REPORT.json
  - publication/tables/sensor_acquisition.tex
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

from sensors.sensor_acquisition import (
    SensorModality,
    DEFAULT_SENSOR_PROFILES,
    AcquisitionPolicy,
    AdaptiveSensorAcquisitionEngine,
)
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp31_sensor_acq")


def run_sensor_acquisition_benchmark(n_events_per_regime: int = 500, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    regimes = [
        {
            "name": "Routine Benign",
            "threat_mean": 0.05,
            "unc_mean": 0.10,
            "criticality": 1.0,
            "cpu_load": 0.20,
        },
        {
            "name": "Ambiguous Anomaly",
            "threat_mean": 0.55,
            "unc_mean": 0.75,
            "criticality": 1.2,
            "cpu_load": 0.40,
        },
        {
            "name": "Critical Attack",
            "threat_mean": 0.90,
            "unc_mean": 0.50,
            "criticality": 2.0,
            "cpu_load": 0.60,
        },
        {
            "name": "Flash Congestion",
            "threat_mean": 0.70,
            "unc_mean": 0.80,
            "criticality": 1.5,
            "cpu_load": 0.95,
        },
    ]

    all_modalities = list(DEFAULT_SENSOR_PROFILES.keys())

    regime_results = []

    for reg in regimes:
        reg_name = reg["name"]
        log.info(f"Evaluating telemetry acquisition under '{reg_name}'...")

        engine = AdaptiveSensorAcquisitionEngine()

        # Metrics trackers for 3 strategies
        # 1. Network Only
        net_gains, net_costs, net_lats, net_mems = [], [], [], []
        # 2. Full Stack
        full_gains, full_costs, full_lats, full_mems = [], [], [], []
        # 3. Adaptive VOI
        voi_gains, voi_costs, voi_lats, voi_mems = [], [], [], []

        for i in range(n_events_per_regime):
            p_threat = float(np.clip(rng.normal(reg["threat_mean"], 0.05), 0.0, 1.0))
            p_unc = float(np.clip(rng.normal(reg["unc_mean"], 0.05), 0.0, 1.0))
            crit = reg["criticality"]
            cpu = reg["cpu_load"]

            # Strategy 1: Network Only
            p_base = DEFAULT_SENSOR_PROFILES[SensorModality.BASELINE_NETWORK]
            net_gains.append(p_base.base_information_gain * (0.6 * p_unc + 0.4 * p_threat))
            net_costs.append(p_base.nominal_cost_units)
            net_lats.append(p_base.nominal_latency_ms)
            net_mems.append(p_base.memory_footprint_mb)

            # Strategy 2: Full Stack (all except unauthorized deep forensics)
            f_gain = sum(
                p.base_information_gain * (0.6 * p_unc + 0.4 * p_threat) * (crit / 1.5)
                for m, p in DEFAULT_SENSOR_PROFILES.items()
                if m != SensorModality.DEEP_FORENSIC
            )
            f_cost = sum(
                p.nominal_cost_units * (1.0 + 1.5 * max(0.0, cpu - 0.5))
                for m, p in DEFAULT_SENSOR_PROFILES.items()
                if m != SensorModality.DEEP_FORENSIC
            )
            f_lat = sum(p.nominal_latency_ms for m, p in DEFAULT_SENSOR_PROFILES.items() if m != SensorModality.DEEP_FORENSIC)
            f_mem = sum(p.memory_footprint_mb for m, p in DEFAULT_SENSOR_PROFILES.items() if m != SensorModality.DEEP_FORENSIC)

            full_gains.append(min(1.0, f_gain * 0.4))
            full_costs.append(f_cost)
            full_lats.append(f_lat)
            full_mems.append(f_mem)

            # Strategy 3: AHRAS Adaptive VOI
            plan = engine.plan_event_telemetry(
                event_id=f"evt-{reg_name[:3]}-{i}",
                threat_prior=p_threat,
                epistemic_uncertainty=p_unc,
                asset_criticality=crit,
                cpu_load=cpu,
            )
            acquired = [d for d in plan if d.should_acquire]
            v_gain = sum(d.expected_security_gain for d in acquired) / max(1, len(acquired))
            v_cost = sum(d.collection_cost for d in acquired)
            v_lat = sum(d.estimated_latency_ms for d in acquired)
            v_mem = sum(DEFAULT_SENSOR_PROFILES[d.modality].memory_footprint_mb for d in acquired)

            voi_gains.append(v_gain)
            voi_costs.append(v_cost)
            voi_lats.append(v_lat)
            voi_mems.append(v_mem)

        regime_results.append({
            "regime": reg_name,
            "network_only": {
                "mean_security_gain": round(float(np.mean(net_gains)), 3),
                "mean_cost_units": round(float(np.mean(net_costs)), 3),
                "mean_latency_ms": round(float(np.mean(net_lats)), 2),
                "mean_memory_mb": round(float(np.mean(net_mems)), 1),
            },
            "full_stack": {
                "mean_security_gain": round(float(np.mean(full_gains)), 3),
                "mean_cost_units": round(float(np.mean(full_costs)), 3),
                "mean_latency_ms": round(float(np.mean(full_lats)), 2),
                "mean_memory_mb": round(float(np.mean(full_mems)), 1),
            },
            "adaptive_voi": {
                "mean_security_gain": round(float(np.mean(voi_gains)), 3),
                "mean_cost_units": round(float(np.mean(voi_costs)), 3),
                "mean_latency_ms": round(float(np.mean(voi_lats)), 2),
                "mean_memory_mb": round(float(np.mean(voi_mems)), 1),
                "cost_savings_vs_full_stack_pct": round((1.0 - float(np.mean(voi_costs)) / float(np.mean(full_costs))) * 100.0, 1),
                "latency_reduction_pct": round((1.0 - float(np.mean(voi_lats)) / float(np.mean(full_lats))) * 100.0, 1),
            },
        })

    return {"regimes_evaluated": regime_results, "events_per_regime": n_events_per_regime}


def export_latex_table(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Adaptive Sensor Acquisition vs Static Baselines across Operational Regimes (EXP-31)}",
        "\\label{tab:sensor_acquisition}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|ccc|ccc|cccc}",
        "\\hline",
        "\\textbf{Operational Regime} & \\multicolumn{3}{c|}{\\textbf{Network-Only Baseline}} & \\multicolumn{3}{c|}{\\textbf{Full-Stack Baseline}} & \\multicolumn{4}{c}{\\textbf{AHRAS Adaptive VOI Acquisition}} \\\\",
        " & Gain & Cost & Latency (ms) & Gain & Cost & Latency (ms) & Gain & Cost & Latency (ms) & Cost Reduction \\\\",
        "\\hline",
    ]

    for r in results:
        reg = r["regime"]
        net = r["network_only"]
        full = r["full_stack"]
        voi = r["adaptive_voi"]
        tex.append(
            f"{reg} & {net['mean_security_gain']:.2f} & {net['mean_cost_units']:.2f} & {net['mean_latency_ms']:.1f} & "
            f"{full['mean_security_gain']:.2f} & {full['mean_cost_units']:.2f} & {full['mean_latency_ms']:.1f} & "
            f"\\textbf{{{voi['mean_security_gain']:.2f}}} & \\textbf{{{voi['mean_cost_units']:.2f}}} & \\textbf{{{voi['mean_latency_ms']:.1f}}} & "
            f"\\textbf{{-{voi['cost_savings_vs_full_stack_pct']:.1f}\\%}} \\\\"
        )

    tex.extend([
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    log.info(f"Exported LaTeX table to {output_path}")


def main() -> None:
    log.info("Starting EXP-31: Adaptive Sensor Acquisition Benchmark...")
    bench = run_sensor_acquisition_benchmark(n_events_per_regime=500)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "SENSOR_ACQUISITION_REPORT.json"

    manifest = create_manifest(
        experiment_id="EXP-31",
        dataset_name="Multi-Regime Telemetry Stream",
        dataset_path="evaluation/data/dataset_registry.json",
        dataset_version="1.0",
        configuration={"events_per_regime": 500, "modalities_count": len(DEFAULT_SENSOR_PROFILES)},
        result_path=str(report_path),
        random_seed=42,
    )

    final_report = {
        "manifest": manifest.to_dict(),
        "benchmark_results": bench["regimes_evaluated"],
        "metadata": {"events_per_regime": bench["events_per_regime"]},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved machine-readable report to {report_path}")

    latex_path = Path(__file__).resolve().parent.parent / "publication" / "tables" / "sensor_acquisition.tex"
    export_latex_table(bench["regimes_evaluated"], latex_path)
    print("\n=== EXP-31 BENCHMARK SUMMARY ===")
    for row in bench["regimes_evaluated"]:
        voi = row["adaptive_voi"]
        print(f"[{row['regime']}] Security Gain: {voi['mean_security_gain']} | Cost: {voi['mean_cost_units']} | Latency: {voi['mean_latency_ms']}ms | Savings vs Full: {voi['cost_savings_vs_full_stack_pct']}%")


if __name__ == "__main__":
    main()
