from __future__ import annotations
"""
AHRAS Phase 5 — Paper Reproducibility & Benchmark Evaluation Suite
-------------------------------------------------------------------
Automates paper benchmark experiments, evaluating detection accuracy (Precision,
Recall, F1-Score), inference latency (ms), and false-positive rates across
all multi-modal threat vectors.

Outputs:
  1. Console Evaluation Summary Table.
  2. Publication LaTeX Table Code (`eval/paper_results_table.tex`) for direct insertion
     into research paper evaluations.
"""

import time
import os
import sys
import logging

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

from detection.dataset_generator import generate_dataset
from detection.feature_extractor import extract
from detection.hybrid_engine import get_combiner, DetectionResult
from detection.pipeline import bootstrap_ml_models
from detection.risk_engine import run_risk_engine
from normalizer.ocsf_normalizer import _norm_network, _norm_process, _norm_file, _norm_cloud

log = logging.getLogger(__name__)


@dataclass
class ScenarioMetrics:
    scenario_name: str
    ocsf_class:    str
    total_events:  int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision:     float
    recall:        float
    f1_score:      float
    avg_latency_ms: float


class PaperBenchmarkEvaluator:
    """
    Runs automated paper evaluation experiments and formats LaTeX results.
    """

    def __init__(self):
        self.combiner = get_combiner()

    def run_all_experiments(self) -> List[ScenarioMetrics]:
        print("=======================================================================")
        print("   AHRAS Paper Reproducibility & Benchmark Evaluation Suite")
        print("=======================================================================")
        print("Bootstrapping hybrid detection engine and ML models...")
        bootstrap_ml_models()
        print("Engine ready. Running multi-modal evaluation scenarios...\n")

        results = []
        scenarios = [
            ("Network: Port Scanning", "network_activity", self._gen_port_scan),
            ("Network: SYN Flood", "network_activity", self._gen_syn_flood),
            ("Network: SSH Brute Force", "network_activity", self._gen_ssh_brute),
            ("Host File: Ransomware Entropy", "file_activity", self._gen_ransomware),
            ("Host Process: Credential Dump", "process_activity", self._gen_cred_dump),
            ("Cloud API: Defense Evasion", "cloud_api", self._gen_cloud_evasion),
        ]

        for name, cls, gen_fn in scenarios:
            metrics = self._evaluate_scenario(name, cls, gen_fn)
            results.append(metrics)

        self._print_summary_table(results)
        self._export_latex_table(results)
        return results

    def _evaluate_scenario(self, name: str, cls: str, gen_fn) -> ScenarioMetrics:
        events, labels = gen_fn()
        tp, fp, tn, fn = 0, 0, 0, 0
        latencies = []

        for evt, is_attack in zip(events, labels):
            t0 = time.perf_counter()
            det_res = self.combiner.process(evt)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000.0)

            is_pred_anom = det_res.is_alert

            if is_attack and is_pred_anom:
                tp += 1
            elif not is_attack and is_pred_anom:
                fp += 1
            elif not is_attack and not is_pred_anom:
                tn += 1
            elif is_attack and not is_pred_anom:
                fn += 1

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)
        avg_lat = float(np.mean(latencies))

        return ScenarioMetrics(
            scenario_name=name, ocsf_class=cls, total_events=len(events),
            true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn,
            precision=round(prec, 4), recall=round(rec, 4), f1_score=round(f1, 4),
            avg_latency_ms=round(avg_lat, 2)
        )

    # ── Synthetic Dataset Generators ─────────────────────────────────────────

    def _gen_port_scan(self) -> Tuple[List[dict], List[bool]]:
        events, labels = [], []
        # Normal (50)
        for i in range(50):
            events.append(_norm_network({"src_ip": f"192.168.1.{i%10+1}", "packet_count": 5, "duration_sec": 2}))
            labels.append(False)
        # Attack (20)
        for i in range(20):
            events.append(_norm_network({"src_ip": "10.0.0.99", "unique_dst_ports": 150, "packet_count": 300, "duration_sec": 1, "tcp_flags": ["SYN"]}))
            labels.append(True)
        return events, labels

    def _gen_syn_flood(self) -> Tuple[List[dict], List[bool]]:
        events, labels = [], []
        for i in range(50):
            events.append(_norm_network({"src_ip": f"192.168.1.{i%10+1}", "packet_count": 10}))
            labels.append(False)
        for i in range(20):
            events.append(_norm_network({"src_ip": "172.16.0.4", "packet_count": 15000, "duration_sec": 1, "tcp_flags": ["SYN"]}))
            labels.append(True)
        return events, labels

    def _gen_ssh_brute(self) -> Tuple[List[dict], List[bool]]:
        events, labels = [], []
        for i in range(50):
            events.append(_norm_network({"src_ip": f"192.168.1.{i%10+1}", "packet_count": 8}))
            labels.append(False)
        for i in range(20):
            events.append(_norm_network({"src_ip": "198.51.100.44", "dst_port": 22, "packet_count": 500, "duration_sec": 3}))
            labels.append(True)
        return events, labels

    def _gen_ransomware(self) -> Tuple[List[dict], List[bool]]:
        events, labels = [], []
        for i in range(50):
            events.append(_norm_file({"filepath": f"/home/user/doc_{i}.pdf", "entropy": 3.2}))
            labels.append(False)
        for i in range(20):
            events.append(_norm_file({"filepath": f"/home/user/data_{i}.locked", "entropy": 7.9, "high_entropy": True}))
            labels.append(True)
        return events, labels

    def _gen_cred_dump(self) -> Tuple[List[dict], List[bool]]:
        events, labels = [], []
        for i in range(50):
            events.append(_norm_process({"name": "chrome.exe", "pid": 100 + i}))
            labels.append(False)
        for i in range(20):
            events.append(_norm_process({"name": "lsass.exe", "cmdline": "mimikatz.exe privilege::debug", "pid": 500 + i}))
            labels.append(True)
        return events, labels

    def _gen_cloud_evasion(self) -> Tuple[List[dict], List[bool]]:
        events, labels = [], []
        for i in range(50):
            events.append(_norm_cloud({"user_identity": f"user_{i}@corp.com", "action": "s3:GetObject"}))
            labels.append(False)
        for i in range(20):
            events.append(_norm_cloud({"user_identity": "attacker@corp.com", "action": "cloudtrail:StopLogging", "severity_hint": "critical"}))
            labels.append(True)
        return events, labels

    # ── Formatting & Latex Output ─────────────────────────────────────────────

    def _print_summary_table(self, results: List[ScenarioMetrics]):
        print("\n" + "─"*95)
        print(f"{'Threat Scenario':<32} | {'Class':<16} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Lat(ms)':<8}")
        print("─"*95)
        for r in results:
            print(f"{r.scenario_name:<32} | {r.ocsf_class:<16} | {r.precision:<6.3f} | {r.recall:<6.3f} | {r.f1_score:<6.3f} | {r.avg_latency_ms:<8.2f}")
        print("─"*95 + "\n")

    def _export_latex_table(self, results: List[ScenarioMetrics]):
        os.makedirs("eval", exist_ok=True)
        path = "eval/paper_results_table.tex"
        latex = []
        latex.append("% AHRAS Publication Results Table")
        latex.append("\\begin{table}[ht]")
        latex.append("\\centering")
        latex.append("\\caption{AHRAS Multi-Modal Threat Detection Performance across OCSF Schema Classes}")
        latex.append("\\label{tab:ahras_results}")
        latex.append("\\begin{tabular}{l l c c c c}")
        latex.append("\\hline")
        latex.append("\\textbf{Threat Scenario} & \\textbf{OCSF Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Latency (ms)} \\\\")
        latex.append("\\hline")
        for r in results:
            latex.append(f"{r.scenario_name} & {r.ocsf_class} & {r.precision:.3f} & {r.recall:.3f} & {r.f1_score:.3f} & {r.avg_latency_ms:.2f} \\\\")
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(path, "w") as f:
            f.write("\n".join(latex))
        print(f"✓ Saved paper LaTeX table code to: {path}")


if __name__ == "__main__":
    evaluator = PaperBenchmarkEvaluator()
    evaluator.run_all_experiments()
