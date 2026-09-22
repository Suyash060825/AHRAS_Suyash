from __future__ import annotations
"""
AHRAS Subsystem — Two-Tier Endpoint Telemetry Adapter & Shannon Entropy Watcher
--------------------------------------------------------------------------------
Implements AHRAS v11 Host Telemetry Subsystem:
  - Tier 1: Kernel Event Buffer Adapter (eBPF / ETW system call stream simulation).
  - Tier 2: Targeted Block-Level Shannon Entropy Watcher & Process Tree Lineage Reasoner.
  - OCSF Normalization:
      Class 1002 (process_activity)
      Class 1003 (file_activity)
      Class 9001 (network_conn)
  - Ultra-low ingestion overhead (target <= 3.0% CPU) via selective entropy hashing.
"""

import os
import sys
import math
import time
import uuid
import socket
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional, Set

import numpy as np

log = logging.getLogger(__name__)

_HOSTNAME = socket.gethostname()
RANSOMWARE_ENTROPY_THRESHOLD = 7.20 # High entropy threshold for encryption (> 7.2 bits/byte)

# Living-Off-The-Land Binaries (LOLBins) and suspicious parent-child chains (MITRE T1059, T1204)
SUSPICIOUS_LINEAGE_RULES = [
    {"parent": "explorer.exe", "child": "powershell.exe", "threat": "User-initiated PowerShell execution"},
    {"parent": "winword.exe",  "child": "cmd.exe",        "threat": "Office macro spawning command shell (T1204)"},
    {"parent": "excel.exe",    "child": "powershell.exe", "threat": "Excel spawning PowerShell (T1204)"},
    {"parent": "cmd.exe",      "child": "certutil.exe",   "threat": "Certificate utility ingress tool (T1105)"},
    {"parent": "cmd.exe",      "child": "wmic.exe",       "threat": "WMI execution (T1047)"},
    {"parent": "powershell.exe","child": "vssadmin.exe",  "threat": "Shadow copy deletion attempt (T1490)"},
    {"parent": "httpd",        "child": "bash",           "threat": "Web server reverse shell execution (T1059)"},
    {"parent": "nginx",        "child": "sh",             "threat": "Web server spawning shell (T1059)"},
    {"parent": "python3",      "child": "nc",             "threat": "Python spawning Netcat reverse shell (T1059)"},
    {"parent": "svchost.exe",  "child": "cmd.exe",        "threat": "Service host spawning interactive shell (T1055)"},
]


# ── Fast Shannon Entropy Calculator ──────────────────────────────────────────

def compute_block_shannon_entropy(data: bytes) -> float:
    """
    Computes Shannon entropy H(X) = -sum P(x_i) * log2(P(x_i)) over a byte buffer.
    Theoretical range: [0.0, 8.0] bits per byte.
    Ransomware encrypted data typically exhibits H(X) > 7.2.
    Plaintext ASCII code typically exhibits H(X) in [3.5, 5.0].
    """
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    n = len(data)
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    return round(float(entropy), 4)


# ── Process Lineage Graph Tracker ─────────────────────────────────────────────

@dataclass
class ProcessNode:
    pid: int
    name: str
    cmdline: str
    parent_pid: Optional[int]
    parent_name: str
    user: str
    timestamp: float
    depth: int = 1


class ProcessLineageTracker:
    """
    Maintains endpoint process execution tree and detects LOLBin / adversary lineage patterns.
    """

    def __init__(self, max_history: int = 5000):
        self.max_history = max_history
        self._tree: Dict[int, ProcessNode] = {}
        self._suspicious_pairings = {
            (r["parent"].lower(), r["child"].lower()): r["threat"] for r in SUSPICIOUS_LINEAGE_RULES
        }

    def register_process(
        self,
        pid: int,
        name: str,
        cmdline: str,
        parent_pid: Optional[int],
        parent_name: str = "",
        user: str = "SYSTEM",
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], int]:
        """
        Registers a process spawn event, updates tree depth, and checks for suspicious lineage.
        Returns: (is_suspicious, threat_description, tree_depth)
        """
        t = timestamp or time.time()
        parent_node = self._tree.get(parent_pid) if parent_pid else None
        depth = (parent_node.depth + 1) if parent_node else 1

        resolved_parent_name = parent_name or (parent_node.name if parent_node else "unknown")
        node = ProcessNode(
            pid=pid,
            name=name,
            cmdline=cmdline,
            parent_pid=parent_pid,
            parent_name=resolved_parent_name,
            user=user,
            timestamp=t,
            depth=depth,
        )
        self._tree[pid] = node

        # Prune if exceeded max history
        if len(self._tree) > self.max_history:
            oldest_pids = sorted(self._tree.keys(), key=lambda k: self._tree[k].timestamp)[:1000]
            for op in oldest_pids:
                self._tree.pop(op, None)

        # Check suspicious pairing rule
        pair = (resolved_parent_name.lower(), name.lower())
        threat = self._suspicious_pairings.get(pair)

        # Additional heuristic: deep subshells spawned by service accounts
        if not threat and depth >= 4 and name.lower() in ("cmd.exe", "powershell.exe", "bash", "sh"):
            threat = f"Abnormal deep subshell execution (depth={depth})"

        is_suspicious = threat is not None
        return is_suspicious, threat, depth

    def get_ancestor_chain(self, pid: int) -> List[str]:
        """Traverses backwards up the process tree."""
        chain = []
        curr_pid = pid
        visited = set()
        while curr_pid and curr_pid not in visited:
            visited.add(curr_pid)
            node = self._tree.get(curr_pid)
            if not node:
                break
            chain.append(f"{node.name}({node.pid})")
            curr_pid = node.parent_pid
        return chain


# ── Two-Tier Telemetry Adapter ────────────────────────────────────────────────

class TwoTierTelemetryAdapter:
    """
    High-performance host telemetry adapter:
      Tier 1: Kernel Event Buffer Ingestion (zero-copy event parsing).
      Tier 2: Conditional Shannon Entropy & Process Lineage Deep Inspection.
    """

    def __init__(self, entropy_threshold: float = RANSOMWARE_ENTROPY_THRESHOLD):
        self.entropy_threshold = entropy_threshold
        self.lineage_tracker = ProcessLineageTracker()
        self.events_processed = 0
        self.high_entropy_events = 0
        self.suspicious_lineages = 0
        self.total_bytes_hashed = 0

    def process_raw_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes raw kernel/host event to OCSF format with two-tier selective inspection.
        """
        t0 = time.perf_counter()
        ev_type = raw_event.get("event_type", "unknown")
        ocsf_event: Dict[str, Any] = {
            "metadata": {
                "event_id": raw_event.get("event_id", str(uuid.uuid4())),
                "timestamp": raw_event.get("timestamp", time.time()),
                "hostname": raw_event.get("hostname", _HOSTNAME),
                "source": "ahras_two_tier_host_agent",
            }
        }

        # ── Tier 1 & Tier 2: File Write & Entropy Inspection ──────────────────
        if ev_type == "file_write":
            ocsf_event["class_uid"] = 1003 # file_activity
            ocsf_event["class_name"] = "file_activity"
            filepath = raw_event.get("filepath", "")
            raw_bytes = raw_event.get("content_sample", b"")

            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("utf-8")

            # Tier 2: Compute Shannon Entropy
            entropy = compute_block_shannon_entropy(raw_bytes)
            is_high_entropy = entropy >= self.entropy_threshold

            # Conditional Hashing: ONLY hash if high entropy or suspicious extension
            file_sha = ""
            if is_high_entropy:
                file_sha = hashlib.sha256(raw_bytes).hexdigest()
                self.total_bytes_hashed += len(raw_bytes)
                self.high_entropy_events += 1

            ocsf_event["file"] = {
                "path": filepath,
                "name": os.path.basename(filepath),
                "extension": os.path.splitext(filepath)[1].lower(),
                "entropy": entropy,
                "is_encrypted_risk": is_high_entropy,
                "sha256": file_sha,
                "size_bytes": len(raw_bytes),
            }

        # ── Tier 1 & Tier 2: Process Spawn & Lineage Inspection ───────────────
        elif ev_type == "process_spawn":
            ocsf_event["class_uid"] = 1002 # process_activity
            ocsf_event["class_name"] = "process_activity"
            pid = int(raw_event.get("pid", 0))
            name = str(raw_event.get("name", "unknown"))
            cmdline = str(raw_event.get("cmdline", ""))
            parent_pid = raw_event.get("parent_pid")
            parent_name = str(raw_event.get("parent_name", ""))
            user = str(raw_event.get("user", "SYSTEM"))

            # Tier 2: Lineage Graph Tracking
            is_susp, threat_desc, depth = self.lineage_tracker.register_process(
                pid=pid,
                name=name,
                cmdline=cmdline,
                parent_pid=parent_pid,
                parent_name=parent_name,
                user=user,
                timestamp=ocsf_event["metadata"]["timestamp"],
            )

            if is_susp:
                self.suspicious_lineages += 1

            ancestors = self.lineage_tracker.get_ancestor_chain(pid)
            ocsf_event["process"] = {
                "pid": pid,
                "name": name,
                "cmdline": cmdline,
                "parent_pid": parent_pid,
                "parent_name": parent_name,
                "lineage_depth": depth,
                "is_suspicious_lineage": is_susp,
                "lineage_threat": threat_desc,
                "ancestor_chain": ancestors,
                "user": user,
            }

        # ── Tier 1: Outbound Network Connection ───────────────────────────────
        elif ev_type == "network_conn":
            ocsf_event["class_uid"] = 1001 # network_activity
            ocsf_event["class_name"] = "network_activity"
            ocsf_event["connection"] = {
                "src_ip": raw_event.get("local_ip", "127.0.0.1"),
                "src_port": raw_event.get("local_port", 0),
                "dst_ip": raw_event.get("remote_ip", "10.0.0.1"),
                "dst_port": raw_event.get("remote_port", 443),
                "protocol": raw_event.get("protocol", "TCP"),
                "pid": raw_event.get("pid"),
                "process_name": raw_event.get("process_name", "unknown"),
            }

        else:
            ocsf_event["class_uid"] = 0
            ocsf_event["class_name"] = "generic_system_event"
            ocsf_event["data"] = raw_event

        t_elapsed = (time.perf_counter() - t0) * 1000.0 # ms
        ocsf_event["metadata"]["normalization_time_ms"] = round(t_elapsed, 4)
        self.events_processed += 1
        return ocsf_event

    def get_adapter_stats(self) -> Dict[str, Any]:
        return {
            "events_processed": self.events_processed,
            "high_entropy_events": self.high_entropy_events,
            "suspicious_lineages": self.suspicious_lineages,
            "total_bytes_hashed": self.total_bytes_hashed,
            "entropy_threshold": self.entropy_threshold,
        }
