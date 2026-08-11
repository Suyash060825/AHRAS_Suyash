from __future__ import annotations
"""
AHRAS Detection Pipeline
-------------------------
Connects Module 1 (telemetry) to Module 2 (detection).

Flow:
  normalized.events topic
        ↓
  HybridCombiner.process(evt)
        ↓
  DetectionResult
        ↓
  storage: alerts collection
        ↓
  detection.results topic  (consumed by Module 4: Risk Engine)

Startup sequence:
  1. Bootstrap ML models with synthetic normal traffic
  2. Start consuming normalized.events
  3. For each event: run hybrid detection
  4. If is_alert: persist + publish to detection.results
  5. Always: update baselines for statistical engine
"""

import json
import time
import logging
import threading
import dataclasses
from datetime import datetime, timezone

import numpy as np

from config.settings import KAFKA_TOPIC_NORM, MONGO_COL_ALERTS
from pipeline.bus import get_producer, get_consumer
from storage.store import get_store
from detection.hybrid_engine import get_combiner, DetectionResult
from detection.anomaly_engine.ml_engine import bootstrap_with_normal_traffic, get_model_status
from detection.dataset_generator import get_normal_vectors, generate_dataset

log = logging.getLogger(__name__)

KAFKA_TOPIC_DETECTIONS = "detection.results"
SUPPORTED_CLASSES      = [
    "network_activity", "process_activity",
    "file_activity", "cloud_api",
]

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ml_models() -> None:
    """
    Pre-train ML models with synthetic normal traffic before going live.
    This ensures the anomaly engine is operational from first event.
    Takes ~10–20 seconds depending on hardware.
    """
    log.info("[PIPELINE] Bootstrapping ML models with synthetic normal data...")
    for cls in SUPPORTED_CLASSES:
        try:
            normals = get_normal_vectors(cls, n=500)
            bootstrap_with_normal_traffic(cls, normals)
            status = get_model_status().get(cls, {})
            log.info(
                f"[PIPELINE] Bootstrap {cls}: "
                f"trained={status.get('trained')}, "
                f"samples={status.get('buffer_size')}"
            )
        except Exception as e:
            log.error(f"[PIPELINE] Bootstrap failed for {cls}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Result serialization
# ─────────────────────────────────────────────────────────────────────────────

def _result_to_dict(r: DetectionResult) -> dict:
    d = dataclasses.asdict(r)
    # Remove the full normalized event from alert record to save space
    # (it's already in the events collection)
    d.pop("normalized_event", None)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stats
# ─────────────────────────────────────────────────────────────────────────────

_stats = {
    "received": 0, "processed": 0, "alerts": 0,
    "errors": 0, "skipped": 0,
}
_stats_lock = threading.Lock()


def get_pipeline_stats() -> dict:
    with _stats_lock:
        s = dict(_stats)
    s.update(get_combiner().get_stats())
    s["model_status"] = get_model_status()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline loop
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_pipeline(stop_event: threading.Event = None) -> None:
    """
    Blocking loop. Consumes normalized.events, runs detection,
    persists alerts. Pass a threading.Event to stop gracefully.
    """
    consumer  = get_consumer(KAFKA_TOPIC_NORM, group_id="ahras-detection")
    producer  = get_producer()
    store     = get_store()
    combiner  = get_combiner()

    log.info("[PIPELINE] Detection pipeline started")
    log.info("[PIPELINE] Consuming: normalized.events")
    log.info("[PIPELINE] Publishing alerts to: detection.results")

    for evt in consumer:
        if stop_event and stop_event.is_set():
            break

        with _stats_lock:
            _stats["received"] += 1

        cls = evt.get("ocsf_class", "")
        if cls not in SUPPORTED_CLASSES + ["network_conn"]:
            with _stats_lock:
                _stats["skipped"] += 1
            continue

        try:
            result = combiner.process(evt)
            if result is None:
                with _stats_lock:
                    _stats["skipped"] += 1
                continue

            with _stats_lock:
                _stats["processed"] += 1

            # Always publish result to detection.results (even non-alerts)
            # Module 4 (risk engine) needs all scored events
            result_dict = _result_to_dict(result)
            producer.send(KAFKA_TOPIC_DETECTIONS, result_dict)

            # Persist alerts to storage
            if result.is_alert:
                store.insert(MONGO_COL_ALERTS, {
                    **result_dict,
                    "event_id": result.event_id,
                })
                with _stats_lock:
                    _stats["alerts"] += 1

                log.warning(
                    f"[ALERT] {result.severity:8s} | "
                    f"{result.attack_type:30s} | "
                    f"conf={result.confidence:.2f} | "
                    f"engines={result.engines_fired} | "
                    f"{result.explanation.get('explanation_text','')[:80]}"
                )
            else:
                log.debug(
                    f"[OK]    {cls:25s} | "
                    f"conf={result.confidence:.2f} | "
                    f"{result.processing_ms:.1f}ms"
                )

        except Exception as e:
            with _stats_lock:
                _stats["errors"] += 1
            log.error(f"[PIPELINE] Detection error: {e}", exc_info=True)


def start_detection_pipeline_thread(bootstrap: bool = True) -> threading.Thread:
    """
    Start the detection pipeline as a background daemon thread.
    bootstrap=True → pre-trains ML models before starting.
    """
    def _run():
        if bootstrap:
            bootstrap_ml_models()
        run_detection_pipeline()

    stop_event = threading.Event()
    t = threading.Thread(
        target=_run,
        name="ahras-detection-pipeline",
        daemon=True,
    )
    t.start()
    log.info("[PIPELINE] Detection pipeline thread started")
    return t
