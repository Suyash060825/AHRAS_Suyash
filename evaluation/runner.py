from __future__ import annotations
"""
AHRAS Evaluation Benchmark Runner
----------------------------------
Executes chronological dataset evaluation across any DatasetLoader instance,
converting raw records into OCSF format and running the full AHRAS detection
and risk scoring pipeline.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional, Any

from evaluation.dataset_loader import DatasetLoader, DatasetRecord
from evaluation.metrics import MetricsCalculator, MetricsReport
from normalizer.ocsf_normalizer import _norm_network
from detection.hybrid_engine import get_combiner, DetectionResult
from detection.risk_engine import run_risk_engine, get_risk_engine

log = logging.getLogger(__name__)


def record_to_ocsf(rec: DatasetRecord) -> dict:
    """Converts a DatasetRecord into an OCSF network_activity event dict."""
    feats = rec.features
    
    # Use standardized keys populated by DatasetLoader, fallback to dataset-specific names
    dst_port = int(feats.get("dst_port", feats.get("Destination Port", 80)))
    packet_count = int(feats.get("packet_count", feats.get("Total Fwd Packets", 1) + feats.get("Total Backward Packets", 0)))
    duration_sec = feats.get("duration_sec", max(0.001, feats.get("Flow Duration", 1000.0) / 1_000_000.0))
    bytes_count = int(feats.get("byte_count", feats.get("Total Length of Fwd Packets", 100)))
    pps = float(feats.get("Flow Packets/s", 10.0))
    if pps <= 0 and duration_sec > 0:
        pps = packet_count / duration_sec
        
    syn_count = feats.get("SYN Flag Count", feats.get("syn_count", 0))
    tcp_flags = ["SYN"] if syn_count > 0 else ["ACK"]
    
    unique_dst_ports = feats.get("unique_dst_ports", 1)
    if unique_dst_ports == 1 and dst_port > 1024 and feats.get("Total Fwd Packets", 0) <= 3:
        unique_dst_ports = 150  # Heuristic for port scanning if raw features indicate

    return _norm_network({
        "src_ip":           rec.src_ip,
        "dst_port":         dst_port,
        "packet_count":     packet_count,
        "duration_sec":     duration_sec,
        "byte_count":       bytes_count,
        "bytes":            bytes_count,
        "pps":              pps,
        "tcp_flags":        tcp_flags,
        "unique_dst_ports": int(unique_dst_ports),
    })


class EvaluationRunner:
    def __init__(self):
        self.combiner = get_combiner()
        self.risk_engine = get_risk_engine()
        self.metrics_calc = MetricsCalculator()

    def run_evaluation(
        self,
        loader: DatasetLoader,
        limit: Optional[int] = 5000,
        threshold: float = 0.50,
    ) -> MetricsReport:
        records = list(loader.iter_records(limit=limit))
        if not records:
            return MetricsReport(dataset_name=loader.dataset_type)

        y_true = []
        y_score = []
        latencies = []
        cats = []

        for rec in records:
            ocsf_evt = record_to_ocsf(rec)
            
            t0 = time.perf_counter()
            det_res = self.combiner.process(ocsf_evt)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            if det_res:
                score = det_res.confidence
            else:
                score = 0.0

            y_true.append(rec.label)
            y_score.append(score)
            cats.append(rec.attack_category)

        report = self.metrics_calc.compute(
            y_true=y_true,
            y_score=y_score,
            latencies_ms=latencies,
            dataset_name=loader.dataset_type,
            threshold=threshold,
            attack_categories=cats,
        )
        return report
