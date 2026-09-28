"""
AHRAS Experiment Runner: EXP-36 — Security Twin Response Lab Benchmark (Sections 30 & 31)
------------------------------------------------------------------------------------------
Evaluates the Security Twin as an isolated counterfactual response lab:
  1. Multi-candidate mitigation simulation across cyber kill-chain attack scenarios.
  2. Evaluates NO_ACTION, BLOCK_SOURCE, ISOLATE_HOST, REVOKE_TOKEN, TERMINATE_PROCESS.
  3. Measures Risk Reduction, Path Breakage Probability, Residual Risk, TTC,
     Blast Radius, Collateral Cost, and Action Reversibility.
  4. Multi-objective Pareto action ranking and dry-run safety verification.

Outputs:
  - evaluation/results/SECURITY_TWIN_RESPONSE_LAB_REPORT.json
  - publication/tables/security_twin_response_lab.tex
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

from security_twin.models import (
    AttackScenario,
    AttackStage,
    AttackStep,
    Host,
    Network,
    Process,
    Service,
    User,
)
from security_twin.state import SecurityTwin
from security_twin.simulation import SecurityTwinSimulator
from evaluation.research_manifest import create_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("exp36_security_twin_response_lab")


def build_enterprise_twin() -> SecurityTwin:
    """Builds realistic enterprise digital twin topology."""
    twin = SecurityTwin("twin-enterprise-eval")
    # Hosts
    twin.add_host(Host(host_id="web-prod-01", hostname="web-prod-01", ip_address="10.0.1.10", criticality=0.6))
    twin.add_host(Host(host_id="app-prod-01", hostname="app-prod-01", ip_address="10.0.1.20", criticality=0.7))
    twin.add_host(Host(host_id="db-prod-01", hostname="db-prod-01", ip_address="10.0.1.50", criticality=0.95))
    twin.add_host(Host(host_id="workstation-42", hostname="ws-42", ip_address="10.0.2.42", criticality=0.3))

    # Users
    twin.add_user(User(user_id="user-svc-web", username="svc_web", role="service", is_privileged=False))
    twin.add_user(User(user_id="user-dba", username="db_admin", role="admin", is_privileged=True))

    # Processes
    twin.add_process(Process(pid=8080, process_name="nginx", host_id="web-prod-01", user_id="user-svc-web"))
    twin.add_process(Process(pid=9001, process_name="gunicorn", host_id="app-prod-01", user_id="user-svc-web"))
    twin.add_process(Process(pid=5432, process_name="postgres", host_id="db-prod-01", user_id="user-dba"))

    # Networks
    twin.add_network(Network(network_id="net-dmz", cidr="10.0.1.0/24"))
    twin.add_network(Network(network_id="net-internal", cidr="10.0.2.0/24"))

    # Services
    twin.add_service(Service(service_id="svc-web", name="web_frontend", host_id="web-prod-01", port=443))
    twin.add_service(Service(service_id="svc-api", name="app_api", host_id="app-prod-01", port=8000))
    twin.add_service(Service(service_id="svc-db", name="database", host_id="db-prod-01", port=5432))

    return twin


def evaluate_response_lab_scenarios() -> Dict[str, Any]:
    """Evaluates multi-candidate counterfactual simulation across 3 concrete attack scenarios."""
    twin = build_enterprise_twin()
    simulator = SecurityTwinSimulator(twin)

    scenarios = [
        {
            "name": "External Web Application Exploit & Lateral Spread",
            "scenario": AttackScenario(
                scenario_id="scen-web-lateral",
                name="External Web Exploit to DB Compromise",
                description="Perimeter exploit followed by lateral database reconnaissance",
                steps=[
                    AttackStep(
                        step_id="step-1",
                        stage=AttackStage.INITIAL_ACCESS,
                        timestamp=100.0,
                        source="198.51.100.99",
                        destination="10.0.1.10",
                        technique="T1190",
                        technique_name="Exploit Public-Facing Application",
                        preconditions={"src_ip": "198.51.100.99", "dst_host": "web-prod-01"},
                        postconditions={"compromised": "web-prod-01"},
                    ),
                    AttackStep(
                        step_id="step-2",
                        stage=AttackStage.LATERAL_MOVEMENT,
                        timestamp=110.0,
                        source="10.0.1.10",
                        destination="10.0.1.50",
                        technique="T1021",
                        technique_name="Remote Services",
                        preconditions={"src_host": "web-prod-01", "dst_host": "db-prod-01"},
                        postconditions={"compromised": "db-prod-01"},
                    ),
                ],
            ),
            "candidates": [
                ("NO_ACTION", "web-prod-01"),
                ("BLOCK_SOURCE", "198.51.100.99"),
                ("ISOLATE_HOST", "web-prod-01"),
                ("TERMINATE_PROCESS", "8080"),
            ],
            "incident_risk": 0.88,
        },
        {
            "name": "Privileged Service Account Compromise & Internal Exfiltration",
            "scenario": AttackScenario(
                scenario_id="scen-priv-exfil",
                name="DB Admin Credential Abuse",
                description="Stolen DBA credentials used to exfiltrate customer records",
                steps=[
                    AttackStep(
                        step_id="step-1",
                        stage=AttackStage.CREDENTIAL_ACCESS,
                        timestamp=200.0,
                        source="10.0.2.42",
                        destination="10.0.1.50",
                        technique="T1078",
                        technique_name="Valid Accounts",
                        preconditions={"src_host": "workstation-42", "dst_host": "db-prod-01"},
                        postconditions={"compromised": "db-prod-01"},
                    ),
                    AttackStep(
                        step_id="step-2",
                        stage=AttackStage.EXFILTRATION,
                        timestamp=220.0,
                        source="10.0.1.50",
                        destination="203.0.113.15",
                        technique="T1048",
                        technique_name="Exfiltration Over Alternative Protocol",
                        preconditions={"src_host": "db-prod-01"},
                        postconditions={"exfiltrated": True},
                    ),
                ],
            ),
            "candidates": [
                ("NO_ACTION", "db-prod-01"),
                ("REVOKE_TOKEN", "user-dba"),
                ("ISOLATE_HOST", "db-prod-01"),
                ("TERMINATE_PROCESS", "5432"),
            ],
            "incident_risk": 0.94,
        },
    ]

    scenario_evaluations = []

    for item in scenarios:
        log.info(f"Simulating Response Lab for scenario: {item['name']}")
        evals = simulator.simulate_candidate_responses(
            item["scenario"],
            item["candidates"],
            current_risk=item["incident_risk"],
        )
        rec = next((e for e in evals if e["is_recommended"]), None)
        scenario_evaluations.append({
            "scenario_name": item["name"],
            "incident_risk": item["incident_risk"],
            "candidate_evaluations": evals,
            "recommended_action": rec["action_type"] if rec else "NONE",
            "recommended_target": rec["target_entity"] if rec else "NONE",
            "recommended_utility": rec["utility_score"] if rec else 0.0,
            "recommended_risk_reduction": rec["risk_reduction"] if rec else 0.0,
            "recommended_path_breakage": rec["path_breakage_probability"] if rec else 0.0,
            "recommended_blast_radius": rec["blast_radius_score"] if rec else 0.0,
            "recommended_reversibility": rec["reversibility"] if rec else 0.0,
        })

    return {
        "scenarios_evaluated": len(scenario_evaluations),
        "results": scenario_evaluations,
    }


def export_latex_table(scenario_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Exports IEEE-format LaTeX table for Security Twin Response Lab evaluations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex = [
        "% Auto-generated by evaluation/run_security_twin_response_lab_eval.py (EXP-36)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Security Twin Counterfactual Response Lab Evaluation (EXP-36)}",
        "\\label{tab:security_twin_response_lab}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Action Candidate} & \\textbf{Pre-Risk} & \\textbf{Post-Risk} & \\textbf{Path Breakage} & \\textbf{Blast Radius} & \\textbf{Reversibility} & \\textbf{Recommended} \\\\",
        "\\hline",
    ]

    for sc in scenario_results:
        sc_name = sc["scenario_name"]
        tex.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{sc_name}}}}} \\\\")
        for c in sc["candidate_evaluations"]:
            tex.append(
                f"{c['action_type']} & "
                f"{c['pre_action_risk']:.2f} & "
                f"{c['expected_post_action_risk']:.2f} & "
                f"{c['path_breakage_probability'] * 100:.0f}\\% & "
                f"{c['blast_radius_score']:.2f} & "
                f"{c['reversibility']:.2f} & "
                f"{'\\textbf{Yes}' if c['is_recommended'] else 'No'} \\\\"
            )
        tex.append("\\hline")

    tex.extend([
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    log.info(f"Exported LaTeX table to {output_path}")


def main() -> None:
    log.info("Starting EXP-36: Security Twin Response Lab Benchmark...")

    eval_data = evaluate_response_lab_scenarios()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "SECURITY_TWIN_RESPONSE_LAB_REPORT.json"

    manifest = create_manifest(
        experiment_id="EXP-36",
        dataset_name="Enterprise Digital Twin Counterfactual Response Lab",
        dataset_path="evaluation/data/dataset_registry.json",
        dataset_version="1.0",
        configuration={
            "scenarios_count": eval_data["scenarios_evaluated"],
            "candidates_per_scenario": 4,
            "dry_run_safety": True,
        },
        result_path=str(report_path),
        random_seed=42,
    )

    final_report = {
        "manifest": manifest.to_dict(),
        "response_lab_evaluation": eval_data,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"Saved machine-readable report to {report_path}")

    latex_path = Path(__file__).resolve().parent.parent / "publication" / "tables" / "security_twin_response_lab.tex"
    export_latex_table(eval_data["results"], latex_path)

    print("\n=== EXP-36 BENCHMARK SUMMARY ===")
    for sc in eval_data["results"]:
        print(f"\nScenario: {sc['scenario_name']}")
        print(f"  Recommended Mitigation: {sc['recommended_action']} on {sc['recommended_target']}")
        print(f"  Utility Score: {sc['recommended_utility']:.4f} | Risk Reduction: {sc['recommended_risk_reduction']:.4f}")
        print(f"  Path Breakage: {sc['recommended_path_breakage']*100:.1f}% | Blast Radius: {sc['recommended_blast_radius']:.4f} | Reversibility: {sc['recommended_reversibility']:.2f}")


if __name__ == "__main__":
    main()
