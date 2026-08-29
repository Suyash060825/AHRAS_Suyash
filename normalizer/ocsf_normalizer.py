"""
AHRAS OCSF Normalizer
----------------------
Consumes raw.telemetry, normalizes every event to OCSF schema,
enriches with GeoIP + AbuseIPDB, then:
  - Publishes to normalized.events topic
  - Writes to storage (SQLite/MongoDB)

OCSF class IDs used:
  1001 → network_activity   (network flows)
  1002 → process_activity   (process spawns)
  1003 → file_activity      (file writes, entropy)
  4001 → cloud_api          (cloud audit logs)
  9001 → network_conn       (per-process connections from host agent)
"""

import uuid
import logging
import threading
from datetime import datetime, timezone

from config.settings import (
    KAFKA_TOPIC_RAW, KAFKA_TOPIC_NORM, KAFKA_GROUP_ID,
    MONGO_COL_EVENTS,
)
from pipeline.bus import get_producer, get_consumer
from storage.store import get_store
from normalizer.enrichment import enrich_ip

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eid(raw: dict) -> str:
    return raw.get("event_id", str(uuid.uuid4()))


# ─────────────────────────────────────────────────────────────────────────────
# Per-source normalizers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(v, default=""):
    return str(v) if v is not None else default

def _safe_int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def _norm_network(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    src_ip = _safe_str(raw.get("src_ip"))
    dst_ip = _safe_str(raw.get("dst_ip"))
    src_enrich = enrich_ip(src_ip)
    dst_enrich = enrich_ip(dst_ip)

    severity = 1
    if src_enrich["is_threat_intel_hit"]:
        severity = 3
    if src_enrich["is_high_risk"]:
        severity = 4

    unique_ports = _safe_int(raw.get("unique_dst_ports"))
    flags = raw.get("tcp_flags", []) if isinstance(raw.get("tcp_flags"), (list, tuple)) else []
    port_scan = (unique_ports > 20 and "SYN" in flags and "ACK" not in flags)

    return {
        "ocsf_class_id": 1001,
        "ocsf_class":    "network_activity",
        "event_id":      _eid(raw),
        "time":          raw.get("timestamp", _now()),
        "severity_id":   severity,
        "src_endpoint": {
            "ip":   src_ip,
            "port": _safe_int(raw.get("src_port")),
            "geo":  src_enrich["geo"],
        },
        "dst_endpoint": {
            "ip":   dst_ip,
            "port": _safe_int(raw.get("dst_port")),
            "geo":  dst_enrich["geo"],
        },
        "protocol": _safe_str(raw.get("protocol"), "OTHER"),
        "traffic": {
            "packets":      _safe_int(raw.get("packet_count")),
            "bytes":        _safe_int(raw.get("byte_count")),
            "duration_sec": _safe_float(raw.get("duration_sec")),
        },
        "tcp_flags":        flags,
        "unique_dst_ports": unique_ports,
        "enrichment": {
            **src_enrich,
            "port_scan_indicator": port_scan,
        },
        "raw_source": "network_tap",
    }


def _norm_process(raw: dict) -> dict:
    parent_name = raw.get("parent_name", "")
    proc_name   = raw.get("name", "")

    from config.settings import SUSPICIOUS_PARENTS, SUSPICIOUS_CHILDREN
    suspicious = (parent_name in SUSPICIOUS_PARENTS
                  and proc_name in SUSPICIOUS_CHILDREN)

    return {
        "ocsf_class_id": 1002,
        "ocsf_class":    "process_activity",
        "event_id":      _eid(raw),
        "time":          raw.get("timestamp", _now()),
        "severity_id":   3 if suspicious else 1,
        "actor": {
            "process": {
                "pid":      raw.get("pid"),
                "name":     proc_name,
                "exe":      raw.get("exe", ""),
                "cmd_line": raw.get("cmdline", ""),
                "user":     {"name": raw.get("username", "")},
            }
        },
        "process": {
            "parent_pid":  raw.get("parent_pid"),
            "parent_name": parent_name,
        },
        "device": {"hostname": raw.get("hostname", "")},
        "enrichment": {
            "suspicious_lineage": suspicious,
        },
        "raw_source": "host_agent",
    }


def _norm_file(raw: dict) -> dict:
    entropy      = float(raw.get("entropy", 0.0))
    high_entropy = raw.get("high_entropy", entropy > 7.2)

    return {
        "ocsf_class_id": 1003,
        "ocsf_class":    "file_activity",
        "event_id":      _eid(raw),
        "time":          raw.get("timestamp", _now()),
        "severity_id":   4 if entropy > 7.5 else (3 if high_entropy else 1),
        "file": {
            "path":   raw.get("filepath", ""),
            "sha256": raw.get("sha256", ""),
        },
        "device": {"hostname": raw.get("hostname", "")},
        "enrichment": {
            "entropy":               entropy,
            "high_entropy":          high_entropy,
            "ransomware_indicator":  entropy > 7.2,
        },
        "raw_source": "host_agent",
    }


def _norm_net_conn(raw: dict) -> dict:
    remote_ip = raw.get("remote_ip", "")
    enrich    = enrich_ip(remote_ip)

    return {
        "ocsf_class_id": 9001,
        "ocsf_class":    "network_conn",
        "event_id":      _eid(raw),
        "time":          raw.get("timestamp", _now()),
        "severity_id":   3 if enrich["is_threat_intel_hit"] else 1,
        "actor": {
            "process": {
                "pid":  raw.get("pid"),
                "name": raw.get("process_name", ""),
            }
        },
        "src_endpoint": {"addr": raw.get("local_addr", "")},
        "dst_endpoint": {
            "addr": raw.get("remote_addr", ""),
            "ip":   remote_ip,
            "port": raw.get("remote_port", 0),
        },
        "protocol":  raw.get("protocol", "TCP"),
        "device":    {"hostname": raw.get("hostname", "")},
        "enrichment": enrich,
        "raw_source": "host_agent",
    }


def _norm_cloud(raw: dict) -> dict:
    _sev_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    src_ip  = raw.get("source_ip", "")
    enrich  = enrich_ip(src_ip)

    severity = max(
        _sev_map.get(raw.get("severity_hint", "low"), 1),
        3 if enrich["is_threat_intel_hit"] else 1,
    )

    return {
        "ocsf_class_id": 4001,
        "ocsf_class":    "cloud_api",
        "event_id":      _eid(raw),
        "time":          raw.get("timestamp", _now()),
        "severity_id":   severity,
        "api": {
            "operation": raw.get("action", ""),
            "error":     raw.get("error_code"),
        },
        "actor": {
            "user": {"name": raw.get("user_identity", "")},
        },
        "src_endpoint": {
            "ip":  src_ip,
            "geo": enrich["geo"],
        },
        "cloud": {
            "provider": raw.get("provider", ""),
            "region":   raw.get("region", ""),
        },
        "enrichment": {
            **enrich,
            "high_privilege_action": raw.get("severity_hint") in ("high", "critical"),
        },
        "raw_source": "cloud_adapter",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

def _route(raw: dict) -> dict | None:
    source     = raw.get("source", "")
    event_type = raw.get("event_type", "")

    if source == "network_tap":
        return _norm_network(raw)

    if source == "host_agent":
        if event_type == "process_spawn":
            return _norm_process(raw)
        if event_type == "file_write":
            return _norm_file(raw)
        if event_type == "network_conn":
            return _norm_net_conn(raw)
        log.debug(f"[NORM] Unknown host event_type: {event_type}")
        return None

    if source == "cloud_adapter":
        return _norm_cloud(raw)

    log.debug(f"[NORM] Unknown source: {source}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main normalizer loop
# ─────────────────────────────────────────────────────────────────────────────

_stats = {"received": 0, "normalized": 0, "dropped": 0, "errors": 0}
_stats_lock = threading.Lock()


def get_stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def run_normalizer(stop_event: threading.Event = None) -> None:
    """
    Blocking loop. Pass a threading.Event to stop gracefully.
    If stop_event is None, runs until KeyboardInterrupt.
    """
    consumer = get_consumer(KAFKA_TOPIC_RAW, group_id=KAFKA_GROUP_ID)
    producer = get_producer()
    store    = get_store()

    log.info("[NORM] Normalizer started — consuming raw.telemetry")

    for raw in consumer:
        if stop_event and stop_event.is_set():
            break

        with _stats_lock:
            _stats["received"] += 1

        try:
            normalized = _route(raw)
            if normalized is None:
                with _stats_lock:
                    _stats["dropped"] += 1
                continue

            # Publish to normalized topic
            producer.send(KAFKA_TOPIC_NORM, normalized)

            # Persist to storage
            store.insert(MONGO_COL_EVENTS, normalized)

            with _stats_lock:
                _stats["normalized"] += 1

            log.debug(
                f"[NORM] {normalized['ocsf_class']:20s} "
                f"sev={normalized['severity_id']} "
                f"src={normalized.get('src_endpoint', {}).get('ip', '-')}"
            )

        except Exception as e:
            with _stats_lock:
                _stats["errors"] += 1
            log.error(f"[NORM] Error processing event: {e} | raw={raw}")


def start_normalizer_thread() -> threading.Thread:
    """Start normalizer as a background daemon thread."""
    stop_event = threading.Event()
    t = threading.Thread(
        target=run_normalizer,
        args=(stop_event,),
        name="ahras-normalizer",
        daemon=True,
    )
    t.start()
    log.info("[NORM] Normalizer thread started")
    return t
