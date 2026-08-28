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
    return _norm_network({
        "src_ip":           rec.src_ip,
        "dst_port":         int(feats.get("Destination Port", 80)),
        "packet_count":     int(feats.get("Total Fwd Packets", 1) + feats.get("Total Backward Packets", 0)),
        "duration_sec":     max(0.001, feats.get("Flow Duration", 1000.0) / 1_000_000.0),
        "bytes":            int(feats.get("Flow Bytes/s", 100.0) * max(0.001, feats.get("Flow Duration", 1000.0) / 1_000_000.0)),
        "pps":              float(feats.get("Flow Packets/s", 10.0)),
        "tcp_flags":        ["SYN"] if feats.get("SYN Flag Count", 0) > 0 else ["ACK"],
        "unique_dst_ports": 150 if feats.get("Destination Port", 0) > 1024 and feats.get("Total Fwd Packets", 0) <= 3 else 1,
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
