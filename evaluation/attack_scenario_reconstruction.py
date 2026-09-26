from __future__ import annotations
"""
AHRAS Experiment Runner: EXP-13 — Provenance Attack Scenario Reconstruction Benchmark
--------------------------------------------------------------------------------------
Evaluates the structure-preserving graph reconstruction quality across multi-stage attack campaigns:
  1. Targeted Enterprise Ransomware Campaign
  2. Bastion Infiltration & Database Exfiltration
  3. Cloud Privilege Escalation & IAM Backdoor

Measures Node F1, Edge F1, Path Completeness, Depth Similarity, and Normalized Structural Distance
under controlled alert missingness (0%, 10%, 25%, 50%) and background benign noise injection.

Generates:
  - evaluation/results/ATTACK_SCENARIO_RECONSTRUCTION.json
  - evaluation/results/table_attack_reconstruction.tex
"""

import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from provenance.models import (
    ProvenanceNodeType, ProvenanceEdgeType, ProvenanceNode, ProvenanceEdge,
    ProvenanceAttackScenario, GraphQualityMetrics
)
from provenance.graph import ProvenanceGraph
from provenance.reconstructor import AttackScenarioReconstructor
from provenance.metrics import compute_comprehensive_graph_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp13_provenance")


def generate_benchmark_scenarios() -> List[Dict[str, Any]]:
    """Builds ground-truth event streams and ground-truth graphs for 3 distinct attack campaigns."""
    t0 = 1700000000.0

    # ── Campaign 1: Enterprise Ransomware Campaign ──
    c1_events = [
        {"event_id": "c1-ev1", "ocsf_class_id": 1001, "time": t0 + 10, "src_ip": "203.0.113.15", "dst_ip": "host-web-01", "dst_port": 80, "enrichment": {"mitre_technique": "T1046", "mitre_name": "Port Scan"}},
        {"event_id": "c1-ev2", "ocsf_class_id": 1001, "time": t0 + 35, "src_ip": "203.0.113.15", "dst_ip": "host-web-01", "dst_port": 80, "enrichment": {"mitre_technique": "T1190", "mitre_name": "Exploit Web App", "is_threat_intel_hit": True}},
        {"event_id": "c1-ev3", "ocsf_class_id": 1002, "time": t0 + 50, "hostname": "host-web-01", "actor": {"process": {"pid": 2048, "name": "sh", "cmd_line": "sh -c payload"}}, "process": {"parent_pid": 1024}, "enrichment": {"suspicious_lineage": True, "mitre_technique": "T1059.004"}},
        {"event_id": "c1-ev4", "ocsf_class_id": 1001, "time": t0 + 80, "src_ip": "host-web-01", "dst_ip": "host-fs-01", "dst_port": 445, "enrichment": {"mitre_technique": "T1021.002", "mitre_name": "SMB Lateral Traversal"}},
        {"event_id": "c1-ev5", "ocsf_class_id": 1003, "time": t0 + 120, "hostname": "host-fs-01", "file": {"path": "/var/shares/financial_2026.db.enc", "entropy": 7.96}, "enrichment": {"mitre_technique": "T1486", "mitre_name": "Data Encrypted"}},
    ]
    c1_path = [["203.0.113.15", "host-web-01", "host-fs-01", "file:host-fs-01:/var/shares/financial_2026.db.enc"]]

    # ── Campaign 2: Bastion Infiltration & Database Exfiltration ──
    c2_events = [
        {"event_id": "c2-ev1", "ocsf_class_id": 1001, "time": t0 + 20, "src_ip": "198.51.100.8", "dst_ip": "host-bastion-01", "dst_port": 22, "enrichment": {"mitre_technique": "T1110.001", "mitre_name": "SSH Password Guessing"}},
        {"event_id": "c2-ev2", "ocsf_class_id": 1002, "time": t0 + 45, "hostname": "host-bastion-01", "actor": {"process": {"pid": 3050, "name": "cat", "cmd_line": "cat /etc/shadow"}}, "process": {"parent_pid": 1100}, "enrichment": {"suspicious_lineage": True, "mitre_technique": "T1003.008"}},
        {"event_id": "c2-ev3", "ocsf_class_id": 1001, "time": t0 + 75, "src_ip": "host-bastion-01", "dst_ip": "host-db-prod-01", "dst_port": 5432, "enrichment": {"mitre_technique": "T1021.004", "mitre_name": "Remote Database Services"}},
        {"event_id": "c2-ev4", "ocsf_class_id": 1002, "time": t0 + 95, "hostname": "host-db-prod-01", "actor": {"process": {"pid": 4120, "name": "tar", "cmd_line": "tar -czf dump.tar.gz"}}, "enrichment": {"mitre_technique": "T1074", "mitre_name": "Data Staged"}},
        {"event_id": "c2-ev5", "ocsf_class_id": 1001, "time": t0 + 130, "src_ip": "host-db-prod-01", "dst_ip": "198.51.100.8", "dst_port": 443, "enrichment": {"mitre_technique": "T1041", "mitre_name": "Exfiltration Over C2"}},
    ]
    c2_path = [["198.51.100.8", "host-bastion-01", "host-db-prod-01", "198.51.100.8"]]

    # ── Campaign 3: Cloud Privilege Escalation & IAM Backdoor ──
    c3_events = [
        {"event_id": "c3-ev1", "ocsf_class_id": 4001, "time": t0 + 15, "actor": {"user": {"name": "usr-dev-svc"}}, "api": {"operation": "GetSessionToken", "service": "sts.amazonaws.com"}, "enrichment": {"mitre_technique": "T1528", "mitre_name": "Steal Application Access Token"}},
        {"event_id": "c3-ev2", "ocsf_class_id": 4001, "time": t0 + 40, "actor": {"user": {"name": "usr-dev-svc"}}, "api": {"operation": "AttachUserPolicy", "service": "iam.amazonaws.com"}, "enrichment": {"mitre_technique": "T1078.004", "mitre_name": "Cloud Accounts Privilege Abuse"}},
        {"event_id": "c3-ev3", "ocsf_class_id": 4001, "time": t0 + 70, "actor": {"user": {"name": "usr-dev-svc"}}, "api": {"operation": "CreateAccessKey", "service": "iam.amazonaws.com"}, "enrichment": {"mitre_technique": "T1098", "mitre_name": "Account Manipulation"}},
        {"event_id": "c3-ev4", "ocsf_class_id": 4001, "time": t0 + 100, "actor": {"user": {"name": "usr-dev-svc"}}, "api": {"operation": "DeleteBucket", "service": "s3.amazonaws.com"}, "enrichment": {"mitre_technique": "T1485", "mitre_name": "Data Destruction"}},
    ]
    c3_path = [
        ["user:usr-dev-svc", "asset:GetSessionToken"],
        ["user:usr-dev-svc", "asset:AttachUserPolicy"],
        ["user:usr-dev-svc", "asset:CreateAccessKey"],
        ["user:usr-dev-svc", "asset:DeleteBucket"],
    ]

    return [
        {"name": "Enterprise Ransomware", "events": c1_events, "critical_paths": c1_path},
        {"name": "Bastion DB Infiltration", "events": c2_events, "critical_paths": c2_path},
        {"name": "Cloud Privilege Escalation", "events": c3_events, "critical_paths": c3_path},
    ]


def generate_benign_noise_events(count: int = 50, base_time: float = 1700000000.0) -> List[Dict[str, Any]]:
    """Generates benign background telemetry events for noise injection testing."""
    noise = []
    rng = random.Random(1337)
    for i in range(count):
        t = base_time + rng.uniform(0.0, 150.0)
        noise.append({
            "event_id": f"noise-ev-{i}",
            "ocsf_class_id": 1001,
            "time": t,
            "src_ip": f"10.0.9.{rng.randint(2, 250)}",
            "dst_ip": f"10.0.1.{rng.randint(2, 250)}",
            "dst_port": 443,
            "protocol": "TCP",
            "enrichment": {},
        })
    return noise


def run_provenance_experiment() -> Dict[str, Any]:
    """Runs the comprehensive EXP-13 benchmark across missingness conditions and noise."""
    log.info("Starting EXP-13 Provenance Attack Scenario Reconstruction Benchmark...")
    start_time = time.time()

    reconstructor = AttackScenarioReconstructor()
    campaigns = generate_benchmark_scenarios()
    noise_pool = generate_benign_noise_events(count=60)

    # Evaluation Conditions: (Label, drop_ratio, add_noise)
    conditions = [
        ("Clean (0% Missing)", 0.0, False),
        ("10% Missing", 0.10, False),
        ("25% Missing", 0.25, False),
        ("50% Missing (Stealth)", 0.50, False),
        ("Noise Injected (+50 Ev)", 0.0, True),
    ]

    results_by_campaign: Dict[str, List[Dict[str, Any]]] = {}
    aggregate_f1_clean = []
    aggregate_f1_stealth = []

    for camp in campaigns:
        c_name = camp["name"]
        events = camp["events"]
        crit_paths = camp["critical_paths"]
        log.info(f"Evaluating campaign: {c_name} ({len(events)} events)...")

        # 1. Establish Ground-Truth Scenario and Graph
        gt_graph = reconstructor.ingest_events(events)
        gt_scenario = reconstructor.reconstruct_scenario(gt_graph, incident_id=f"gt-{c_name}")
        results_by_campaign[c_name] = []

        for cond_name, drop_ratio, add_noise in conditions:
            rng = random.Random(42)

            # Apply drop ratio to events (always retain at least 2 events)
            if drop_ratio > 0.0:
                k = max(2, int(len(events) * (1.0 - drop_ratio)))
                sampled_events = rng.sample(events, k)
                sampled_events.sort(key=lambda e: e["time"])
            else:
                sampled_events = list(events)

            # Inject noise if enabled
            if add_noise:
                sampled_events.extend(noise_pool[:40])
                sampled_events.sort(key=lambda e: e["time"])

            # 2. Ingest and Reconstruct
            pred_graph = reconstructor.ingest_events(sampled_events)
            pred_scenario = reconstructor.reconstruct_scenario(pred_graph, incident_id=f"pred-{c_name}")

            # 3. Compute Structure-Preserving Graph Metrics
            metrics: GraphQualityMetrics = compute_comprehensive_graph_quality(
                pred_scenario=pred_scenario,
                gt_scenario=gt_scenario,
                pred_graph=pred_graph,
                gt_graph=gt_graph,
                gt_critical_paths=crit_paths,
            )

            record = {
                "condition": cond_name,
                "drop_ratio": drop_ratio,
                "noise_injected": add_noise,
                "reconstructed_stages": pred_scenario.stages,
                "missing_steps_detected": pred_scenario.missing_steps,
                "metrics": metrics.to_dict(),
            }
            results_by_campaign[c_name].append(record)

            if cond_name == "Clean (0% Missing)":
                aggregate_f1_clean.append(metrics.overall_graph_f1)
            elif cond_name == "50% Missing (Stealth)":
                aggregate_f1_stealth.append(metrics.overall_graph_f1)

    elapsed_time = round(time.time() - start_time, 2)
    log.info(f"EXP-13 Benchmark completed in {elapsed_time}s.")

    mean_clean_f1 = float(sum(aggregate_f1_clean) / len(aggregate_f1_clean))
    mean_stealth_f1 = float(sum(aggregate_f1_stealth) / len(aggregate_f1_stealth))

    final_report = {
        "experiment_id": "EXP-13",
        "title": "Provenance Attack Scenario Reconstruction & Graph Quality Benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "runtime_seconds": elapsed_time,
        "summary_metrics": {
            "mean_overall_graph_f1_clean": round(mean_clean_f1, 4),
            "mean_overall_graph_f1_stealth_50pct": round(mean_stealth_f1, 4),
            "campaigns_evaluated": len(campaigns),
            "conditions_per_campaign": len(conditions),
        },
        "results_by_campaign": results_by_campaign,
    }

    # Save JSON report
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ATTACK_SCENARIO_RECONSTRUCTION.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved JSON report to {json_path}")

    # Generate LaTeX table
    tex_path = os.path.join(out_dir, "table_attack_reconstruction.tex")
    generate_latex_table(results_by_campaign, tex_path)
    log.info(f"Saved LaTeX table to {tex_path}")

    return final_report


def generate_latex_table(results: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
    """Compiles LaTeX table summarizing structure/depth-preserving graph quality metrics."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{AHRAS Heterogeneous Provenance Graph Reconstruction Quality (EXP-13)}",
        r"\label{tab:provenance_reconstruction_eval}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Attack Campaign} & \textbf{Telemetry Condition} & \textbf{Node F1} & \textbf{Edge F1} & \textbf{Path Compl.} & \textbf{Depth Sim.} & \textbf{Struct. Dist.} & \textbf{Overall F1} \\",
        r"\midrule",
    ]

    for c_name, entries in results.items():
        first = True
        for row in entries:
            c_col = c_name if first else ""
            first = False
            cond_col = row["condition"]
            m = row["metrics"]
            node_f1 = f"{m['node_f1']:.3f}"
            edge_f1 = f"{m['edge_f1']:.3f}"
            p_comp = f"{m['path_completeness']*100:.1f}\\%"
            d_sim = f"{m['depth_similarity']*100:.1f}\\%"
            s_dist = f"{m['structural_distance']:.3f}"
            ov_f1 = f"{m['overall_graph_f1']:.3f}"

            lines.append(f"{c_col} & {cond_col} & {node_f1} & {edge_f1} & {p_comp} & {d_sim} & {s_dist} & {ov_f1} \\\\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rep = run_provenance_experiment()
    print("\n" + "=" * 80)
    print("EXP-13 PROVENANCE RECONSTRUCTION BENCHMARK COMPLETE")
    print(f"Mean Graph F1 (Clean): {rep['summary_metrics']['mean_overall_graph_f1_clean']:.4f}")
    print(f"Mean Graph F1 (50% Missingness): {rep['summary_metrics']['mean_overall_graph_f1_stealth_50pct']:.4f}")
    print("=" * 80)
