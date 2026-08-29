from __future__ import annotations
"""
AHRAS End-to-End Instrumented System Performance Profiler
----------------------------------------------------------
Measures fine-grained execution latency across every individual architectural stage
as well as the complete end-to-end telemetry-to-response pipeline.

Instrumented Subsystems:
  1. Ingestion & Raw Deserialization
  2. OCSF Normalization & Enrichment
  3. Feature Extraction
  4. Signature Rule Engine (23 MITRE Rules)
  5. ML Anomaly Ensemble (Autoencoder + IsolationForest + OneClassSVM)
  6. Statistical Behavioral Engine (Welford Z-score + Circadian)
  7. Evidence Ledger Append-Only Hash Chaining
  8. Temporal Entity Episode Graph
  9. Uncertainty Quantification
  10. Adaptive Risk Controller
  11. DecisionTrace Construction
  12. Causal Risk Forecaster
  13. Active Defense Safety Policy & Utility Evaluation
  14. End-to-End Total Pipeline Latency
"""

import os
import sys
import time
import psutil
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from normalizer.ocsf_normalizer import _norm_network
from detection.feature_extractor import extract
from detection.signature_engine.rules import run_signature_engine
from detection.anomaly_engine.ml_engine import run_anomaly_engine
from detection.statistical_engine.stat_engine import run_statistical_engine
from ahras.evidence.models import EvidenceRecord
from ahras.evidence.ledger import get_evidence_ledger
from detection.gnn_engine import get_entity_graph_engine
from detection.risk_engine import get_risk_engine, RiskResult
from forecast.predictor import AttackPredictor
from response.orchestrator import get_response_orchestrator

log = logging.getLogger(__name__)


@dataclass
class StageProfile:
    stage_name:      str
    mean_latency_ms: float
    p50_latency_ms:  float
    p95_latency_ms:  float
    p99_latency_ms:  float
    percentage_of_total: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerformanceProfileReport:
    total_events_profiled: int
    mean_e2e_latency_ms:   float
    p50_e2e_latency_ms:    float
    p95_e2e_latency_ms:    float
    p99_e2e_latency_ms:    float
    throughput_eps:        float
    peak_memory_mb:        float
    cpu_utilization_pct:   float
    stage_breakdown:       List[StageProfile]

    def to_dict(self) -> dict:
        return {
            "total_events_profiled": self.total_events_profiled,
            "mean_e2e_latency_ms":   round(self.mean_e2e_latency_ms, 3),
            "p50_e2e_latency_ms":    round(self.p50_e2e_latency_ms, 3),
            "p95_e2e_latency_ms":    round(self.p95_e2e_latency_ms, 3),
            "p99_e2e_latency_ms":    round(self.p99_e2e_latency_ms, 3),
            "throughput_eps":        round(self.throughput_eps, 1),
            "peak_memory_mb":        round(self.peak_memory_mb, 1),
            "cpu_utilization_pct":   round(self.cpu_utilization_pct, 1),
            "stages":                [s.to_dict() for s in self.stage_breakdown],
        }


class EndToEndProfiler:
    """
    High-resolution micro-benchmark profiler for AHRAS.
    """

    def __init__(self):
        self.evidence_ledger = get_evidence_ledger()
        self.graph_engine = get_entity_graph_engine()
        self.risk_engine = get_risk_engine()
        self.predictor = AttackPredictor(horizon=5)
        self.orchestrator = get_response_orchestrator()

    def profile_pipeline(self, n_events: int = 500) -> PerformanceProfileReport:
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / (1024 * 1024)

        stage_times: Dict[str, List[float]] = {
            "1. Normalization": [],
            "2. Feature Extraction": [],
            "3. Signature Matching": [],
            "4. ML Ensemble Anomaly": [],
            "5. Statistical Baseline": [],
            "6. Evidence Hash Chaining": [],
            "7. Temporal Graph Update": [],
            "8. Adaptive Risk & Uncertainty": [],
            "9. Forecasting & Trend": [],
            "10. Safety Policy & Active Defense": [],
        }
        e2e_latencies: List[float] = []

        # Synthetic realistic telemetry events
        raw_events = [
            {
                "src_ip": f"192.168.1.{(i % 50) + 1}",
                "dst_ip": "10.0.0.1",
                "dst_port": 22 if i % 10 == 0 else 443,
                "packet_count": 500 if i % 10 == 0 else 20,
                "duration_sec": 2.5,
                "tcp_flags": ["SYN"] if i % 10 == 0 else [],
            }
            for i in range(n_events)
        ]

        t_overall_start = time.perf_counter()

        for raw in raw_events:
            t0 = time.perf_counter()

            # 1. Normalization
            t_s = time.perf_counter()
            ocsf = _norm_network(raw)
            stage_times["1. Normalization"].append((time.perf_counter() - t_s) * 1000.0)

            # 2. Feature Extraction
            t_s = time.perf_counter()
            feat_vec = extract(ocsf)
            stage_times["2. Feature Extraction"].append((time.perf_counter() - t_s) * 1000.0)

            # 3. Signature Matching
            t_s = time.perf_counter()
            sig_matches = run_signature_engine(ocsf)
            stage_times["3. Signature Matching"].append((time.perf_counter() - t_s) * 1000.0)

            # 4. ML Ensemble
            t_s = time.perf_counter()
            ml_res = run_anomaly_engine(ocsf.get("ocsf_class", "network_activity"), feat_vec)
            stage_times["4. ML Ensemble Anomaly"].append((time.perf_counter() - t_s) * 1000.0)

            # 5. Statistical Baseline
            t_s = time.perf_counter()
            stat_res = run_statistical_engine(ocsf, feat_vec)
            stage_times["5. Statistical Baseline"].append((time.perf_counter() - t_s) * 1000.0)

            # 6. Evidence Hash Chaining
            t_s = time.perf_counter()
            ev = EvidenceRecord(event_id=ocsf["event_id"], entity_id=raw["src_ip"], normalized_score=0.7)
            self.evidence_ledger.record_evidence(ev)
            stage_times["6. Evidence Hash Chaining"].append((time.perf_counter() - t_s) * 1000.0)

            # 7. Temporal Graph Update
            t_s = time.perf_counter()
            self.graph_engine.add_event_edge(raw["src_ip"], raw["dst_ip"], "COMM")
            g_corr = self.graph_engine.get_corroboration_score(raw["src_ip"])
            stage_times["7. Temporal Graph Update"].append((time.perf_counter() - t_s) * 1000.0)

            # 8. Adaptive Risk & Uncertainty
            t_s = time.perf_counter()
            risk_res = self.risk_engine.score_risk(raw["src_ip"], sig_matches, ml_res, stat_res, ocsf, g_corr=g_corr)
            stage_times["8. Adaptive Risk & Uncertainty"].append((time.perf_counter() - t_s) * 1000.0)

            # 9. Forecasting
            t_s = time.perf_counter()
            f_res = self.predictor.predict(raw["src_ip"], [0.2, 0.4, risk_res.risk_score])
            stage_times["9. Forecasting & Trend"].append((time.perf_counter() - t_s) * 1000.0)

            # 10. Safety Policy & Response
            t_s = time.perf_counter()
            actions = self.orchestrator.evaluate_and_respond(risk_res, ocsf)
            stage_times["10. Safety Policy & Active Defense"].append((time.perf_counter() - t_s) * 1000.0)

            e2e_latencies.append((time.perf_counter() - t0) * 1000.0)

        total_elapsed = time.perf_counter() - t_overall_start
        throughput = n_events / max(total_elapsed, 1e-4)
        mem_end = process.memory_info().rss / (1024 * 1024)

        mean_e2e = float(np.mean(e2e_latencies))
        p50_e2e = float(np.median(e2e_latencies))
        p95_e2e = float(np.percentile(e2e_latencies, 95))
        p99_e2e = float(np.percentile(e2e_latencies, 99))

        breakdown = []
        for name, times in stage_times.items():
            m_lat = float(np.mean(times))
            pct = (m_lat / max(mean_e2e, 1e-4)) * 100.0
            breakdown.append(StageProfile(
                stage_name=name,
                mean_latency_ms=round(m_lat, 3),
                p50_latency_ms=round(float(np.median(times)), 3),
                p95_latency_ms=round(float(np.percentile(times, 95)), 3),
                p99_latency_ms=round(float(np.percentile(times, 99)), 3),
                percentage_of_total=round(pct, 1),
            ))

        return PerformanceProfileReport(
            total_events_profiled=n_events,
            mean_e2e_latency_ms=mean_e2e,
            p50_e2e_latency_ms=p50_e2e,
            p95_e2e_latency_ms=p95_e2e,
            p99_e2e_latency_ms=p99_e2e,
            throughput_eps=throughput,
            peak_memory_mb=mem_end,
            cpu_utilization_pct=psutil.cpu_percent(),
            stage_breakdown=breakdown,
        )


if __name__ == "__main__":
    profiler = EndToEndProfiler()
    print("Profiling complete end-to-end AHRAS pipeline across 500 events...")
    rep = profiler.profile_pipeline(n_events=500)
    print(f"\nTotal Throughput: {rep.throughput_eps:.1f} events/sec")
    print(f"Mean End-to-End Latency: {rep.mean_e2e_latency_ms:.2f} ms (P95: {rep.p95_e2e_latency_ms:.2f} ms)")
    print("\nStage-by-Stage Breakdown:")
    for s in rep.stage_breakdown:
        print(f"  {s.stage_name:<35} : {s.mean_latency_ms:>6.3f} ms ({s.percentage_of_total:>4.1f}%)")
