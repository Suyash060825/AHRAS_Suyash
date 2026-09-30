"""
Tests for Semantic Adversarial Mutation & Evasion Robustness Engine (Research Frontier C / EXP-24)
--------------------------------------------------------------------------------------------------
Validates:
- ProcessMutator strategies (case alternation, carets, quoting, padding, paths, flags, aliases)
- TrafficMutator strategies (packet padding, timing jitter, duration dilation, port variation)
- CloudMutator strategies (user-agent, regions, parameter permutations)
- MutationEngine semantic integrity enforcement and perturbation batch generation
- EvasionEvaluator detector robustness scoring (R_det) and strategy ranking
- End-to-end artifact generation and JSON/LaTeX export
"""

import json
from pathlib import Path
import pytest

from adversarial.process_mutator import ProcessMutator, ProcessMutationStrategy
from adversarial.traffic_mutator import TrafficMutator, TrafficMutationStrategy
from adversarial.cloud_mutator import CloudMutator, CloudMutationStrategy
from adversarial.mutation_engine import MutationEngine, AdversarialPerturbationBatch
from adversarial.evasion_evaluator import (
    EvasionEvaluator,
    DetectorRobustnessScore,
    EvasionEvaluationReport,
)
from detection_coverage.implementation_catalog import get_default_catalog
from evaluation.run_evasion_robustness import run_benchmark


class TestProcessMutator:
    def test_all_process_strategies(self):
        mutator = ProcessMutator(seed=42)
        base_event = {
            "ocsf_class": "process_activity",
            "actor": {
                "process": {
                    "name": "powershell.exe",
                    "cmd_line": "powershell.exe -enc aWV4IChjZ2V0KQ==",
                }
            },
        }

        # Case alternation
        res_case = mutator.mutate_event(base_event, ProcessMutationStrategy.CASE_ALTERNATION)
        assert res_case.mutated_cmd != res_case.original_cmd
        assert res_case.mutated_cmd.lower() == res_case.original_cmd.lower()

        # Caret insertion
        res_caret = mutator.mutate_event(base_event, ProcessMutationStrategy.CARET_INSERTION)
        assert "^" in res_caret.mutated_cmd

        # Quote insertion
        res_quote = mutator.mutate_event(base_event, ProcessMutationStrategy.QUOTE_INSERTION)
        assert '""' in res_quote.mutated_cmd

        # Whitespace padding
        res_pad = mutator.mutate_event(base_event, ProcessMutationStrategy.WHITESPACE_PADDING)
        assert "   " in res_pad.mutated_cmd

        # Path variation
        res_path = mutator.mutate_event(base_event, ProcessMutationStrategy.PATH_VARIATION)
        assert "System32" in res_path.mutated_cmd or "WindowsPowerShell" in res_path.mutated_cmd

        # Flag reordering
        flag_event = {
            "ocsf_class": "process_activity",
            "actor": {
                "process": {
                    "name": "vssadmin.exe",
                    "cmd_line": "vssadmin.exe delete shadows /all /quiet",
                }
            }
        }
        res_flags = mutator.mutate_event(flag_event, ProcessMutationStrategy.FLAG_REORDERING)
        assert "/quiet" in res_flags.mutated_cmd and "/all" in res_flags.mutated_cmd


class TestTrafficMutator:
    def test_traffic_mutations(self):
        mutator = TrafficMutator(seed=42)
        flow_event = {
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 80},
            "traffic": {
                "bytes": 5000,
                "packets": 50,
                "pps": 1000.0,
                "duration_sec": 0.05,
                "interval_variance": 0.01,
            },
            "inter_arrival_times": [0.01, 0.01, 0.01, 0.01],
            "packet_lengths": [128, 128, 128],
        }

        # Packet padding
        res_pad = mutator.mutate_event(flow_event, TrafficMutationStrategy.PACKET_PADDING, intensity=0.5)
        assert res_pad.mutated_event["traffic"]["bytes"] > flow_event["traffic"]["bytes"]

        # Timing jitter
        res_jit = mutator.mutate_event(flow_event, TrafficMutationStrategy.TIMING_JITTER, intensity=0.5)
        assert res_jit.mutated_event["traffic"]["interval_variance"] > flow_event["traffic"]["interval_variance"]

        # Duration dilation
        res_dur = mutator.mutate_event(flow_event, TrafficMutationStrategy.DURATION_DILATION, intensity=0.8)
        assert res_dur.mutated_event["traffic"]["duration_sec"] > flow_event["traffic"]["duration_sec"]
        assert res_dur.mutated_event["traffic"]["pps"] < flow_event["traffic"]["pps"]

        # Port variation
        res_port = mutator.mutate_event(flow_event, TrafficMutationStrategy.PORT_VARIATION)
        assert res_port.mutated_event["dst_endpoint"]["port"] == 8080


class TestCloudMutator:
    def test_cloud_mutations(self):
        mutator = CloudMutator(seed=42)
        cloud_event = {
            "ocsf_class": "cloud_api",
            "api": {"operation": "iam:AttachUserPolicy"},
            "actor": {"user": {"name": "admin_service"}},
        }

        res_ua = mutator.mutate_event(cloud_event, CloudMutationStrategy.USER_AGENT_ROTATION)
        assert "user_agent" in res_ua.mutated_event.get("http_request", {})

        res_reg = mutator.mutate_event(cloud_event, CloudMutationStrategy.REGION_SPRAYING)
        assert "region" in res_reg.mutated_event.get("cloud", {})

        res_perm = mutator.mutate_event(cloud_event, CloudMutationStrategy.PERMISSION_TRICKLING)
        assert res_perm.mutated_event["api"]["operation"] == "iam:PutUserPolicy"


class TestMutationEngine:
    def test_semantic_integrity_and_batching(self):
        engine = MutationEngine(seed=42)
        catalog = get_default_catalog()

        # Find process implementation
        p_impl = [i for i in catalog.get_all_implementations() if i.implementation_id == "T1059.004-IMPL-01"][0]
        batch = engine.generate_perturbations_for_implementation(p_impl)

        assert batch.implementation_id == "T1059.004-IMPL-01"
        assert len(batch.variants) >= 5
        # First variant is baseline at epsilon 0.0
        assert batch.variants[0].epsilon_budget == 0.0
        assert batch.variants[0].strategy_name == "baseline_identity"

        for v in batch.variants:
            assert engine.validate_semantic_integrity(v.mutated_event) is True


class TestEvasionEvaluator:
    def test_evaluator_metrics(self):
        evaluator = EvasionEvaluator()
        report = evaluator.run_evaluation()

        assert report.total_evaluated_vectors > 0
        assert report.total_mutation_trials > 0
        assert 0.0 <= report.overall_baseline_recall <= 1.0
        assert 0.0 <= report.overall_mutated_recall <= 1.0
        assert 0.0 <= report.overall_evasion_rate <= 1.0

        # Check detector robustness scores
        engine_names = [d.engine_name for d in report.detector_scores]
        assert "signature" in engine_names
        assert "hybrid" in engine_names

        for d in report.detector_scores:
            assert 0.0 <= d.robustness_score <= 1.0
            assert 0.0 <= d.evasion_rate <= 1.0

        # Check strategy ranking
        assert len(report.strategy_efficacies) > 0
        top_strat = report.strategy_efficacies[0]
        assert top_strat.total_trials > 0

        # Check robustness envelope monotonicity
        for i in range(1, len(report.robustness_envelope)):
            prev = report.robustness_envelope[i - 1]
            curr = report.robustness_envelope[i]
            assert curr.epsilon >= prev.epsilon
            assert curr.mean_uncertainty >= prev.mean_uncertainty


class TestEvasionArtifacts:
    def test_end_to_end_benchmark_run(self):
        report = run_benchmark()
        assert report["experiment_id"] == "EXP-24"
        assert "summary" in report
        assert "detector_scores" in report
        assert "strategy_efficacies" in report

        json_path = Path("evaluation/results/EVASION_ROBUSTNESS_REPORT.json")
        latex_path = Path("publication/tables/evasion_robustness.tex")

        assert json_path.exists()
        assert latex_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["experiment_id"] == "EXP-24"

        with open(latex_path, "r", encoding="utf-8") as f:
            latex = f.read()
        assert r"\begin{table*}" in latex
        assert r"Adversarial Evasion Robustness" in latex
