"""
AHRAS Experiment Runner: EXP-35 — Cryptographic Decision Provenance Benchmark (Section 27)
------------------------------------------------------------------------------------------
Evaluates the Cryptographic Decision Provenance Epoch Ledger:
  1. Append Throughput & Latency across 2,000 automated intervention decisions.
  2. Merkle Root Computation & Checkpointing Latency scaling across epoch capacities.
  3. End-to-End Cryptographic Audit Verification Speed.
  4. Section 27.1 Tamper Detection Rate across 100 adversarial modification trials.

Outputs:
  - evaluation/results/DECISION_PROVENANCE_REPORT.json
  - publication/tables/decision_provenance.tex
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

from ahras.evidence.decision_provenance import (
    GENESIS_PREV_EPOCH_HASH,
    DecisionProvenanceRecord,
    EpochProvenanceLedger,
    compute_merkle_root,
)
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp35_decision_provenance")


def benchmark_append_and_checkpointing(
    n_decisions: int = 2000,
    epoch_capacities: List[int] = [25, 50, 100, 200],
) -> Dict[str, Any]:
    """Evaluates throughput, latency, and scaling across different epoch capacities."""
    capacity_results = []

    for cap in epoch_capacities:
        ledger = EpochProvenanceLedger(epoch_capacity=cap)
        append_latencies = []

        t0 = time.perf_counter()
        for i in range(n_decisions):
            rec = DecisionProvenanceRecord(
                decision_id=f"dec-bench-{cap}-{i:05d}",
                event_hash=f"{i:064x}",
                model_hash="c38ddd6e7214ca197e713732050cbe2a812015d23864511f3e77ed6888a4a02",
                config_hash="76bb13b6e7214ca197e713732050cbe276bb13b6e7214ca197e713732050cbe2",
                policy_version="ahras-policy-v2.1-production",
                risk_trace={
                    "composite_risk": 0.85 + (i % 10) * 0.01,
                    "confidence": 0.94,
                    "epistemic_uncertainty": 0.06,
                },
                xai_hash=f"xai-{i:060x}",
                simulation_hash=f"sim-{i:060x}",
                response_action="CONTAIN_HOST_ISOLATE_VLAN" if i % 2 == 0 else "APPLY_DYNAMIC_FW_RULE",
                timestamp=1700000000.0 + i * 0.05,
            )
            t_rec_0 = time.perf_counter()
            ledger.append_decision(rec)
            t_rec_1 = time.perf_counter()
            append_latencies.append((t_rec_1 - t_rec_0) * 1e6)  # microseconds

        total_elapsed = time.perf_counter() - t0
        # Checkpoint any remaining records
        final_epoch = ledger.checkpoint_epoch()

        throughput = n_decisions / total_elapsed
        lat_arr = np.array(append_latencies)

        # Audit time
        t_audit_0 = time.perf_counter()
        audit_res = ledger.verify_entire_ledger()
        audit_time_ms = (time.perf_counter() - t_audit_0) * 1000.0

        capacity_results.append({
            "epoch_capacity": cap,
            "total_decisions": n_decisions,
            "total_epochs": ledger.total_epochs,
            "total_time_seconds": round(total_elapsed, 4),
            "throughput_decisions_per_sec": round(throughput, 2),
            "append_latency_mean_us": round(float(np.mean(lat_arr)), 2),
            "append_latency_p50_us": round(float(np.percentile(lat_arr, 50)), 2),
            "append_latency_p95_us": round(float(np.percentile(lat_arr, 95)), 2),
            "append_latency_p99_us": round(float(np.percentile(lat_arr, 99)), 2),
            "audit_valid": audit_res["valid"],
            "audit_time_ms": round(audit_time_ms, 2),
            "audit_rate_decisions_per_sec": round(n_decisions / (audit_time_ms / 1000.0), 2),
        })

    return {
        "n_decisions_evaluated": n_decisions,
        "capacity_benchmarks": capacity_results,
    }


def benchmark_tamper_detection(n_trials: int = 100) -> Dict[str, Any]:
    """
    Executes Section 27.1 Tamper Test across 100 trials:
    Alters records in past epochs across various fields and confirms 100% detection rate.
    """
    ledger = EpochProvenanceLedger(epoch_capacity=50)
    for i in range(500):
        rec = DecisionProvenanceRecord(
            decision_id=f"dec-tamper-{i:04d}",
            event_hash=f"{i:064x}",
            model_hash="c38ddd6e7214ca197e713732050cbe2a812015d23864511f3e77ed6888a4a02",
            config_hash="76bb13b6e7214ca197e713732050cbe276bb13b6e7214ca197e713732050cbe2",
            policy_version="ahras-policy-v2.1",
            risk_trace={"composite_risk": 0.88, "confidence": 0.95, "epistemic_uncertainty": 0.05},
            xai_hash=f"xai-{i:060x}",
            simulation_hash=f"sim-{i:060x}",
            response_action="CONTAIN_HOST",
            timestamp=1700000000.0 + i,
        )
        ledger.append_decision(rec)

    # 10 full epochs created
    assert ledger.total_epochs == 10

    fields_to_test = [
        ("event_hash", "malicious_event_digest_" + "0" * 41),
        ("model_hash", "rogue_backdoored_model_" + "0" * 41),
        ("config_hash", "modified_system_config_" + "0" * 41),
        ("policy_version", "unauthorized_policy_v9.9"),
        ("composite_risk", 0.05),
        ("xai_hash", "falsified_explanation_" + "0" * 42),
        ("simulation_hash", "tampered_simulation_" + "0" * 43),
        ("response_action", "ALLOW_TRAFFIC_SILENTLY"),
    ]

    detected_count = 0
    field_detection_breakdown: Dict[str, Dict[str, Any]] = {}
    rng = np.random.RandomState(42)
    detection_latencies_ms = []

    for trial_idx in range(n_trials):
        target_epoch_id = int(rng.randint(0, ledger.total_epochs))
        field_name, tampered_val = fields_to_test[trial_idx % len(fields_to_test)]

        t0 = time.perf_counter()
        detected = ledger.tamper_detect_test(target_epoch_id, field_name, tampered_val)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        detection_latencies_ms.append(dt_ms)

        if detected:
            detected_count += 1

        if field_name not in field_detection_breakdown:
            field_detection_breakdown[field_name] = {"trials": 0, "detected": 0}
        field_detection_breakdown[field_name]["trials"] += 1
        if detected:
            field_detection_breakdown[field_name]["detected"] += 1

    detection_rate_pct = (detected_count / n_trials) * 100.0

    return {
        "trials_evaluated": n_trials,
        "epochs_in_ledger": ledger.total_epochs,
        "decisions_in_ledger": ledger.total_decisions,
        "tamper_detected_count": detected_count,
        "tamper_detection_rate_pct": round(detection_rate_pct, 4),
        "mean_detection_latency_ms": round(float(np.mean(detection_latencies_ms)), 3),
        "p95_detection_latency_ms": round(float(np.percentile(detection_latencies_ms, 95)), 3),
        "field_breakdown": field_detection_breakdown,
        "section_27_1_verified": detection_rate_pct == 100.0,
    }


def export_latex_table(benchmarks: List[Dict[str, Any]], tamper_res: Dict[str, Any], output_path: Path) -> None:
    """Exports IEEE-format LaTeX table summarizing Decision Provenance performance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = [
        "% Auto-generated by evaluation/run_decision_provenance_eval.py (EXP-35)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cryptographic Decision Provenance Epoch Ledger Performance (EXP-35)}",
        "\\label{tab:decision_provenance}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "\\textbf{Epoch Capacity} & \\textbf{Throughput (dec/s)} & \\textbf{P50 Lat. ($\\mu$s)} & \\textbf{P95 Lat. ($\\mu$s)} & \\textbf{Audit Rate (dec/s)} & \\textbf{Tamper Det. Rate} \\\\",
        "\\hline",
    ]

    for b in benchmarks:
        tex.append(
            f"{b['epoch_capacity']} records & "
            f"{b['throughput_decisions_per_sec']:,.1f} & "
            f"{b['append_latency_p50_us']:.1f} & "
            f"{b['append_latency_p95_us']:.1f} & "
            f"{b['audit_rate_decisions_per_sec']:,.1f} & "
            f"{tamper_res['tamper_detection_rate_pct']:.1f}\\% \\\\"
        )

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
    log.info("Starting EXP-35: Cryptographic Decision Provenance Benchmark...")

    bench_res = benchmark_append_and_checkpointing(n_decisions=2000, epoch_capacities=[25, 50, 100, 200])
    tamper_res = benchmark_tamper_detection(n_trials=100)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "DECISION_PROVENANCE_REPORT.json"

    manifest = create_manifest(
        experiment_id="EXP-35",
        dataset_name="AHRAS High-Impact Automated Decision Provenance Epoch Ledger",
        dataset_path="evaluation/data/dataset_registry.json",
        dataset_version="1.0",
        configuration={
            "n_decisions": 2000,
            "epoch_capacities": [25, 50, 100, 200],
            "tamper_trials": 100,
            "merkle_hash_algo": "SHA-256",
        },
        result_path=str(report_path),
        random_seed=42,
    )

    final_report = {
        "manifest": manifest.to_dict(),
        "append_and_checkpointing": bench_res,
        "tamper_detection_audit": tamper_res,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved machine-readable report to {report_path}")

    latex_path = Path(__file__).resolve().parent.parent / "publication" / "tables" / "decision_provenance.tex"
    export_latex_table(bench_res["capacity_benchmarks"], tamper_res, latex_path)

    print("\n=== EXP-35 BENCHMARK SUMMARY ===")
    print(f"Decisions Benchmarked: {bench_res['n_decisions_evaluated']}")
    for b in bench_res["capacity_benchmarks"]:
        print(f"  Capacity {b['epoch_capacity']:3d}: {b['throughput_decisions_per_sec']:8.1f} dec/s | P50: {b['append_latency_p50_us']}µs | P95: {b['append_latency_p95_us']}µs | Audit: {b['audit_rate_decisions_per_sec']:8.1f} dec/s")
    print(f"Section 27.1 Tamper Detection Rate: {tamper_res['tamper_detection_rate_pct']}% ({tamper_res['tamper_detected_count']}/{tamper_res['trials_evaluated']} trials)")
    print(f"Section 27.1 Verification Verified: {tamper_res['section_27_1_verified']}")


if __name__ == "__main__":
    main()
