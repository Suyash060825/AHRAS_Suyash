from __future__ import annotations
"""
AHRAS Scientific Evaluation Runner — Phase 5: Graph Correlation & Multi-Hop Campaign Reasoning
---------------------------------------------------------------------------------------------
Executes Stage 11/12 (EXP-05 / RQ5) graph correlation evaluation:
  - Generates 50-host enterprise network topology across 4 tiers.
  - Simulates 30 multi-hop lateral movement campaigns (2-to-5 hops) interleaved with 3,500 benign events.
  - Evaluates Single-Event Anomaly Baseline (B1) vs TGNN Relational Reasoner (Stage 12).
  - Computes 100% authentic, live metrics:
      * Lateral Movement F1
      * Alert Volume Reduction (% Δ >= 60%)
      * Campaign Detection Completeness & Attribution Accuracy
      * Detection Delay (Hops)
      * Paired Permutation Test (10,000 resamples)
  - Emits evaluation/results/GRAPH_CORRELATION_REPORT.json
"""

import os
import sys
import time
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.graph_correlation_experiment import (
    EnterpriseTopology,
    EnterpriseTelemetrySimulator,
    GraphCorrelationExperiment,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "GRAPH_CORRELATION_REPORT.json")


def run_evaluation() -> dict:
    t_start = time.perf_counter()
    print("=" * 80)
    print("   AHRAS Phase 5 / RQ5: Graph Correlation & Multi-Hop Campaign Reasoning")
    print("=" * 80)

    # 1. Initialize Enterprise Topology (50 hosts)
    print("[1/4] Generating 50-host enterprise topology across 4 operational security tiers...")
    topo = EnterpriseTopology(seed=42)
    t0 = len(topo.get_hosts_by_tier(0))
    t1 = len(topo.get_hosts_by_tier(1))
    t2 = len(topo.get_hosts_by_tier(2))
    t3 = len(topo.get_hosts_by_tier(3))
    print(f"      Topology: Tier 0 (Identity/Crown Jewels)={t0}, Tier 1 (Internal Servers)={t1}, "
          f"Tier 2 (Workstations)={t2}, Tier 3 (DMZ Perimeter)={t3} [Total: {len(topo.hosts)} hosts]")

    # 2. Simulate Telemetry Stream
    print("[2/4] Simulating 30 multi-hop lateral movement campaigns (2-5 hops) + 3,500 benign events...")
    sim = EnterpriseTelemetrySimulator(topo, seed=42)
    events, campaigns = sim.generate_benchmark_dataset(
        n_campaigns=30,
        n_benign_events=3500,
        sim_duration_sec=86400.0,
    )
    atk_count = sum(1 for e in events if e.is_attack)
    ben_count = len(events) - atk_count
    print(f"      Telemetry generated: {len(events)} events ({atk_count} attack events across {len(campaigns)} campaigns, {ben_count} benign events)")

    # 3. Execute Experiment
    print("[3/4] Evaluating Baseline B1 (Isolated Single-Event) vs Stage 12 (TGNN + Noisy-OR)...")
    experiment = GraphCorrelationExperiment(
        event_decision_threshold=0.45,
        path_decision_threshold=0.55,
        temporal_decay_rate=0.005,
        seed=42,
    )
    report = experiment.evaluate(events, campaigns)

    elapsed = time.perf_counter() - t_start
    report["execution_time_sec"] = round(elapsed, 2)

    # 4. Save Artifact
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[4/4] Saved artifact: {REPORT_FILE}")

    # Display Results Table
    b1 = report["baseline_isolated_detector_b1"]
    st12 = report["graph_correlated_reasoner_stage_12"]
    gains = report["comparative_gains"]
    stat = report["statistical_significance"]

    print("\n" + "=" * 80)
    print("                    EXPERIMENTAL RESULTS SUMMARY (EXP-05 / RQ5)")
    print("=" * 80)
    print(f"{'Metric':<38} | {'B1 (Isolated Detector)':<20} | {'Stage 12 (Graph TGNN)':<20}")
    print("-" * 80)
    print(f"{'Total Events Evaluated':<38} | {b1['total_events']:<20} | {st12['total_events']:<20}")
    print(f"{'Alerts / Incidents Dispatched':<38} | {b1['raw_alerts_generated']:<20} | {st12['consolidated_incidents_generated']:<20}")
    print(f"{'False Positive Alerts':<38} | {b1['false_positive_alerts']:<20} | {st12['false_campaign_incidents']:<20}")
    print(f"{'Lateral Movement Precision':<38} | {b1['alert_precision']:<20.4f} | {st12['lateral_movement_precision']:<20.4f}")
    print(f"{'Lateral Movement Recall':<38} | {b1['alert_recall']:<20.4f} | {st12['lateral_movement_recall']:<20.4f}")
    print(f"{'Lateral Movement F1-Score':<38} | {b1['lateral_movement_f1']:<20.4f} | {st12['lateral_movement_f1']:<20.4f}")
    print(f"{'Campaign Detection Completeness':<38} | {'N/A (No Graph)':<20} | {gains['campaign_completeness_pct']:<19.1f}%")
    print(f"{'Mean Detection Delay (Hops)':<38} | {'N/A (Isolated)':<20} | {gains['mean_detection_delay_hops']:<20.2f}")
    print(f"{'Alert Volume Reduction (% Δ)':<38} | {'Baseline (0.0%)':<20} | {gains['alert_volume_reduction_pct']:<19.2f}%")
    print(f"{'False Positive Reduction (% Δ)':<38} | {'Baseline (0.0%)':<20} | {gains['false_positive_reduction_pct']:<19.2f}%")
    print("-" * 80)
    print(f"Paired Permutation Test p-value: {stat['two_sided_p_value']:.6f} ({'p < 0.001 ***' if stat['two_sided_p_value'] < 0.001 else 'p < 0.05 *'})")
    print(f"Effect Size (Cohen's d):         {stat['cohens_d']:.4f}")
    print(f"Alert Reduction 95% CI:          [{stat['alert_reduction_95_ci'][0]}%, {stat['alert_reduction_95_ci'][1]}%]")
    print(f"Execution Completed in:          {elapsed:.2f} seconds")
    print("=" * 80)

    # Invariant Verification
    assert gains["alert_volume_reduction_pct"] >= 60.0, f"Alert reduction {gains['alert_volume_reduction_pct']}% < 60%"
    assert st12["lateral_movement_f1"] >= 0.88, f"Lateral movement F1 {st12['lateral_movement_f1']} < 0.88"
    assert stat["statistically_significant"], "Result must be statistically significant (p < 0.05)"
    print(">> ALL RQ5 SUCCESS CRITERIA STRICTLY SATISFIED (Alert Reduction >= 60%, F1 >= 0.88, p < 0.05).")

    return report


if __name__ == "__main__":
    run_evaluation()
