from __future__ import annotations
"""
AHRAS Experiment Runner: EXP-12 — Security Twin Counterfactual & Attack Replay Lab
----------------------------------------------------------------------------------
Rigorously evaluates the Digital Security Twin across multi-stage attack scenarios:
  1. Targeted Enterprise Ransomware Burst
  2. Bastion Compromise & Database Lateral Movement
  3. Privileged Cloud Token Theft & Exfiltration

Compares mitigation policies (NO_ACTION, BLOCK_SOURCE, ISOLATE_HOST, REVOKE_TOKEN, TERMINATE_PROCESS)
via deterministic counterfactual state replay and Monte Carlo uncertainty sampling (N=500 iterations).

Generates:
  - evaluation/results/SECURITY_TWIN_EVALUATION.json
  - evaluation/results/table_security_twin.tex
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from security_twin.models import (
    ActionType, AttackScenario, SimulationResult, MonteCarloResult, SimulationStatus
)
from security_twin.state import SecurityTwin, create_enterprise_test_twin
from security_twin.scenario import (
    build_ransomware_burst_scenario,
    build_lateral_movement_scenario,
    build_credential_abuse_scenario,
)
from security_twin.simulation import SecurityTwinSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp12_security_twin")


def run_security_twin_experiment(num_mc_iterations: int = 500, seed: int = 42) -> Dict[str, Any]:
    """Runs the comprehensive EXP-12 benchmark suite and compiles empirical findings."""
    log.info("Starting EXP-12 Security Twin Counterfactual & Attack Replay Benchmark...")
    start_time = time.time()

    twin = create_enterprise_test_twin()
    simulator = SecurityTwinSimulator(twin)

    scenarios = [
        ("Ransomware Burst", build_ransomware_burst_scenario(
            attacker_ip="203.0.113.55",
            web_host_id="host-web-01",
            file_host_id="host-file-01",
        )),
        ("Lateral Movement", build_lateral_movement_scenario(
            attacker_ip="198.51.100.12",
            bastion_host_id="host-bastion-01",
            db_host_id="host-db-prod-01",
        )),
        ("Credential Abuse", build_credential_abuse_scenario(
            user_id="usr-ops-99",
            token_id="tok-jwt-ops-99",
            cloud_host_id="host-cloud-mgmt",
        )),
    ]

    # Matrix of actions to test per scenario
    action_matrix = {
        "Ransomware Burst": [
            ("NO_ACTION", "none", 0.85),
            ("BLOCK_SOURCE", "203.0.113.55", 0.85),
            ("ISOLATE_HOST", "host-web-01", 0.85),
            ("TERMINATE_PROCESS", "1024", 0.85),
        ],
        "Lateral Movement": [
            ("NO_ACTION", "none", 0.80),
            ("BLOCK_SOURCE", "198.51.100.12", 0.80),
            ("ISOLATE_HOST", "host-bastion-01", 0.80),
            ("ISOLATE_HOST", "host-db-prod-01", 0.80),
        ],
        "Credential Abuse": [
            ("NO_ACTION", "none", 0.82),
            ("REVOKE_TOKEN", "tok-jwt-ops-99", 0.82),
            ("ISOLATE_HOST", "host-cloud-mgmt", 0.82),
        ],
    }

    results_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    aggregate_containment_optimal = []
    aggregate_risk_reduction_optimal = []

    for sc_name, scenario in scenarios:
        log.info(f"Evaluating scenario: {sc_name} ({len(scenario.steps)} kill-chain steps)...")
        results_by_scenario[sc_name] = []
        best_post_risk = 1.0

        for action_type, target, init_risk in action_matrix[sc_name]:
            log.info(f"  -> Testing candidate action: {action_type} on target '{target}'...")

            # 1. Deterministic Counterfactual Replay
            sim_res: SimulationResult = simulator.simulate_action(
                scenario=scenario,
                action_type=action_type,
                target_entity=target,
                current_risk=init_risk,
            )

            # 2. Monte Carlo Uncertainty Sampling
            mc_res: MonteCarloResult = simulator.run_monte_carlo(
                scenario=scenario,
                action_type=action_type,
                target_entity=target,
                num_iterations=num_mc_iterations,
                seed=seed,
            )

            record = {
                "action_type": action_type,
                "target_entity": target,
                "pre_action_risk": sim_res.pre_action_risk,
                "expected_post_action_risk": sim_res.expected_post_action_risk,
                "path_breakage_probability": sim_res.path_breakage_probability,
                "remaining_attack_steps": sim_res.remaining_attack_steps,
                "blast_radius_score": sim_res.blast_radius_score,
                "collateral_disruption_cost": sim_res.collateral_disruption_cost,
                "validation_status": sim_res.validation_status,
                "time_to_containment": sim_res.time_to_containment,
                "explanation": sim_res.explanation,
                "monte_carlo": {
                    "num_iterations": mc_res.num_iterations,
                    "mean_residual_risk": mc_res.mean_residual_risk,
                    "median_residual_risk": mc_res.median_residual_risk,
                    "std_residual_risk": mc_res.std_residual_risk,
                    "p10_risk": mc_res.p10_risk,
                    "p50_risk": mc_res.p50_risk,
                    "p90_risk": mc_res.p90_risk,
                    "p99_risk": mc_res.p99_risk,
                    "containment_probability": mc_res.containment_probability,
                    "escalation_probability": mc_res.escalation_probability,
                    "blast_radius_mean": mc_res.blast_radius_mean,
                },
            }
            results_by_scenario[sc_name].append(record)

            if action_type != "NO_ACTION" and sim_res.expected_post_action_risk < best_post_risk:
                best_post_risk = sim_res.expected_post_action_risk

        # Collect optimal policy containment metric
        optimal_entry = min(
            [r for r in results_by_scenario[sc_name] if r["action_type"] != "NO_ACTION"],
            key=lambda x: x["expected_post_action_risk"]
        )
        aggregate_containment_optimal.append(optimal_entry["monte_carlo"]["containment_probability"])
        red_rate = (optimal_entry["pre_action_risk"] - optimal_entry["expected_post_action_risk"]) / optimal_entry["pre_action_risk"]
        aggregate_risk_reduction_optimal.append(red_rate)

    elapsed_time = round(time.time() - start_time, 2)
    log.info(f"EXP-12 Benchmark completed in {elapsed_time}s.")

    final_report = {
        "experiment_id": "EXP-12",
        "title": "Digital Security Twin & Counterfactual Attack Replay Evaluation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
        "runtime_seconds": elapsed_time,
        "config": {
            "mc_iterations": num_mc_iterations,
            "seed": seed,
            "evaluation_mode": "PRE_EXECUTION_COUNTERFACTUAL_SIMULATION",
        },
        "summary_metrics": {
            "mean_containment_probability_optimal": round(float(sum(aggregate_containment_optimal) / len(aggregate_containment_optimal)), 4),
            "mean_risk_reduction_rate_optimal": round(float(sum(aggregate_risk_reduction_optimal) / len(aggregate_risk_reduction_optimal)), 4),
            "scenarios_evaluated": len(scenarios),
            "total_action_simulations": sum(len(v) for v in results_by_scenario.values()),
        },
        "results_by_scenario": results_by_scenario,
    }

    # Save JSON report
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "SECURITY_TWIN_EVALUATION.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved JSON report to {json_path}")

    # Generate LaTeX Table
    tex_path = os.path.join(out_dir, "table_security_twin.tex")
    generate_latex_table(results_by_scenario, tex_path)
    log.info(f"Saved LaTeX table to {tex_path}")

    return final_report


def generate_latex_table(results: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
    """Generates publication-quality LaTeX table summarizing counterfactual and Monte Carlo outcomes."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{AHRAS Digital Security Twin Response Counterfactual \& Monte Carlo Evaluation (EXP-12)}",
        r"\label{tab:security_twin_eval}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Attack Scenario} & \textbf{Candidate Mitigation} & \textbf{Pre-Risk} & \textbf{Post-Risk} & \textbf{Path Break} & \textbf{Blast Radius} & \textbf{P(Contain)} & \textbf{Status} \\",
        r"\midrule",
    ]

    for sc_name, entries in results.items():
        first = True
        for row in entries:
            sc_col = sc_name if first else ""
            first = False
            act_col = f"{row['action_type']} ({row['target_entity'][:12]})"
            pre_r = f"{row['pre_action_risk']:.2f}"
            post_r = f"{row['expected_post_action_risk']:.2f}"
            p_break = f"{row['path_breakage_probability']*100:.0f}\\%"
            blast = f"{row['blast_radius_score']:.2f}"
            p_cont = f"{row['monte_carlo']['containment_probability']*100:.1f}\\%"
            status = row['validation_status']

            lines.append(f"{sc_col} & {act_col} & {pre_r} & {post_r} & {p_break} & {blast} & {p_cont} & {status} \\\\")
        lines.append(r"\midrule")

    # Remove trailing midrule and add bottomrule
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
    rep = run_security_twin_experiment(num_mc_iterations=500, seed=42)
    print("\n" + "=" * 80)
    print("EXP-12 SECURITY TWIN BENCHMARK COMPLETE")
    print(f"Mean Containment Probability (Optimal Policy): {rep['summary_metrics']['mean_containment_probability_optimal']*100:.2f}%")
    print(f"Mean Risk Reduction Rate (Optimal Policy): {rep['summary_metrics']['mean_risk_reduction_rate_optimal']*100:.2f}%")
    print("=" * 80)
