"""
AHRAS Host Agent
-----------------
Monitors the local endpoint and publishes three event types:

  process_spawn  → process lineage (parent → child)
  file_write     → file modifications with Shannon entropy
  network_conn   → active outbound connections per process

Designed to run as a background daemon. Requires no elevated
privileges for basic operation; root gives fuller process visibility.

Shannon entropy formula:
    H(X) = -Σ P(xᵢ) · log₂ P(xᵢ)
    Range 0–8 bits; encrypted/ransomware data is typically > 7.2
"""

import uuid
import math
import time
import socket
import hashlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.settings import (
    KAFKA_TOPIC_RAW, WATCH_DIRS, ENTROPY_THRESHOLD,
    PROCESS_POLL_SEC, SUSPICIOUS_PARENTS, SUSPICIOUS_CHILDREN,
)
from pipeline.bus import get_producer

log = logging.getLogger(__name__)

_HOSTNAME = socket.gethostname()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Shannon Entropy
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(filepath: str, read_bytes: int = 65536) -> float:
    """
    Compute H(X) = -Σ P(xᵢ) log₂ P(xᵢ) on the first `read_bytes` of a file.
    Returns 0.0 on access errors.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read(read_bytes)
        if not data:
            return 0.0
        freq = [0] * 256
        for byte in data:
            freq[byte] += 1
        n = len(data)
        h = -sum((c / n) * math.log2(c / n) for c in freq if c)
        return round(h, 4)
    except (PermissionError, FileNotFoundError, IsADirectoryError, OSError):
        return 0.0


def file_sha256(filepath: str) -> str:
    """SHA-256 of a file for hash reputation lookups. Empty string on error."""
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (PermissionError, FileNotFoundError, IsADirectoryError, OSError):
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# File system watcher
# ─────────────────────────────────────────────────────────────────────────────

# Extensions commonly targeted by ransomware
_RANSOMWARE_TARGET_EXT = {
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pdf", ".jpg", ".jpeg", ".png", ".mp4", ".zip",
    ".tar", ".gz", ".sql", ".db", ".bak", ".txt",
}

# Extensions added by known ransomware families
_RANSOMWARE_EXT_PATTERNS = {
    ".enc", ".locked", ".crypt", ".crypto", ".crypted",
    ".encrypted", ".pay2me", ".cry", ".wnry", ".wncry",
}


class _FileEventHandler(FileSystemEventHandler):
    def __init__(self, producer):
        super().__init__()
        self._producer = producer

    def on_modified(self, event):
        if not event.is_directory:
            self._process(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._process(event.src_path)

    def _process(self, filepath: str) -> None:
        try:
            path = Path(filepath)
            entropy = shannon_entropy(filepath)
            high_entropy = entropy >= ENTROPY_THRESHOLD

            # Compute hash only for high-entropy files (avoid hashing everything)
            sha256 = file_sha256(filepath) if high_entropy else ""

            ext = path.suffix.lower()
            ransomware_ext = ext in _RANSOMWARE_EXT_PATTERNS
            targeted_ext   = ext in _RANSOMWARE_TARGET_EXT

            event_record = {
                "event_id":           str(uuid.uuid4()),
                "source":             "host_agent",
                "event_type":         "file_write",
                "timestamp":          _now_iso(),
                "hostname":           _HOSTNAME,
                "filepath":           filepath,
                "extension":          ext,
                "entropy":            entropy,
                "high_entropy":       high_entropy,
                "ransomware_ext":     ransomware_ext,
                "targeted_ext":       targeted_ext,
                "sha256":             sha256,
            }

            self._producer.send(KAFKA_TOPIC_RAW, event_record)

            if high_entropy:
                log.warning(
                    f"[HOST] HIGH ENTROPY file: {filepath} "
                    f"(H={entropy:.4f})"
                    + (" [RANSOMWARE EXT]" if ransomware_ext else "")
                )
        except Exception as e:
            log.debug(f"[HOST] File event error for {filepath}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Process monitor
# ─────────────────────────────────────────────────────────────────────────────

class _ProcessMonitor:
    def __init__(self, producer):
        self._producer = producer
        self._seen_pids: set = set(psutil.pids())   # seed on start

    def poll(self) -> None:
        current_pids = set(psutil.pids())
        new_pids     = current_pids - self._seen_pids
        self._seen_pids = current_pids

        for pid in new_pids:
            try:
                proc   = psutil.Process(pid)
                parent = proc.parent()
                p_name = proc.name()
                par_name = parent.name() if parent else ""

                suspicious = (par_name in SUSPICIOUS_PARENTS
                              and p_name in SUSPICIOUS_CHILDREN)

                record = {
                    "event_id":    str(uuid.uuid4()),
                    "source":      "host_agent",
                    "event_type":  "process_spawn",
                    "timestamp":   _now_iso(),
                    "hostname":    _HOSTNAME,
                    "pid":         pid,
                    "name":        p_name,
                    "exe":         _safe(proc.exe),
                    "cmdline":     " ".join(_safe(proc.cmdline, [])[:30]),
                    "username":    _safe(proc.username, ""),
                    "parent_pid":  parent.pid if parent else None,
                    "parent_name": par_name,
                    "suspicious_lineage": suspicious,
                }

                self._producer.send(KAFKA_TOPIC_RAW, record)

                if suspicious:
                    log.warning(
                        f"[HOST] SUSPICIOUS LINEAGE: {par_name}"
                        f"(pid={parent.pid}) → {p_name}(pid={pid})"
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def _safe(fn, default=None):
    """Call a psutil method safely, returning default on error."""
    try:
        return fn() if callable(fn) else fn
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Connection monitor
# ─────────────────────────────────────────────────────────────────────────────

class _ConnectionMonitor:
    def __init__(self, producer):
        self._producer = producer

    def poll(self) -> None:
        try:
            conns = psutil.net_connections(kind="inet")
        except psutil.AccessDenied:
            log.debug("[HOST] Cannot read connections — try running as root")
            return

        for conn in conns:
            if conn.status != "ESTABLISHED" or not conn.raddr:
                continue
            try:
                proc = psutil.Process(conn.pid) if conn.pid else None
                record = {
                    "event_id":     str(uuid.uuid4()),
                    "source":       "host_agent",
                    "event_type":   "network_conn",
                    "timestamp":    _now_iso(),
                    "hostname":     _HOSTNAME,
                    "pid":          conn.pid,
                    "process_name": proc.name() if proc else "unknown",
                    "local_addr":   f"{conn.laddr.ip}:{conn.laddr.port}",
                    "remote_addr":  f"{conn.raddr.ip}:{conn.raddr.port}",
                    "remote_ip":    conn.raddr.ip,
                    "remote_port":  conn.raddr.port,
                    "protocol":     "TCP" if conn.type == 1 else "UDP",
                }
                self._producer.send(KAFKA_TOPIC_RAW, record)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Host Agent orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class HostAgent:
    def __init__(self):
        self._producer = get_producer()
        self._observer = Observer()
        self._proc_mon = _ProcessMonitor(self._producer)
        self._conn_mon = _ConnectionMonitor(self._producer)
        self._running  = False

    def start(self, stop_event: threading.Event = None) -> None:
        # Set up file system watchers
        handler = _FileEventHandler(self._producer)
        watched = 0
        for d in WATCH_DIRS:
            p = Path(d.strip())
            if p.exists():
                self._observer.schedule(handler, str(p), recursive=True)
                log.info(f"[HOST] Watching: {p}")
                watched += 1
        if watched == 0:
            log.warning("[HOST] No watch directories found — file monitoring disabled")

        self._observer.start()
        self._running = True
        log.info(f"[HOST] Agent started on {_HOSTNAME}")

        try:
            while self._running:
                if stop_event and stop_event.is_set():
                    break
                self._proc_mon.poll()
                self._conn_mon.poll()
                time.sleep(PROCESS_POLL_SEC)
        except KeyboardInterrupt:
            pass
        finally:
            self._observer.stop()
            self._observer.join()
            log.info("[HOST] Agent stopped")

    def stop(self) -> None:
        self._running = False
        self._observer.stop()


def start_host_agent_thread() -> threading.Thread:
    """Start host agent as a background daemon thread."""
    stop_event = threading.Event()
    agent = HostAgent()
    t = threading.Thread(
        target=agent.start,
        args=(stop_event,),
        name="ahras-host-agent",
        daemon=True,
    )
    t.start()
    return t
