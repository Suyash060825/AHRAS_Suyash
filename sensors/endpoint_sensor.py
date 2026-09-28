from __future__ import annotations
"""
AHRAS Endpoint Sensor & Telemetry Collector (Section 37)
---------------------------------------------------------
Captures and normalizes host-level behavioral telemetry across:
  - Process creation & lineage (eBPF / Linux auditd / Windows ETW)
  - File modifications, rename bursts & entropy calculation
  - Network socket connections & fan-out monitoring
  - Authentication events & privilege escalations
  - Persistence-related system modifications

Architecture:
  - Kernel / sensor abstractions keep expensive analytics in user space.
  - Generates standardized EndpointEvent records ready for behavioral detection
    and multimodal representation fusion.
"""

import uuid
import time
import math
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class EndpointEventType(str, Enum):
    PROCESS_SPAWN   = "process_spawn"
    FILE_OPERATION  = "file_operation"
    NETWORK_CONNECT = "network_connect"
    AUTH_EVENT      = "auth_event"
    PRIVILEGE_CHANGE = "privilege_change"
    PERSISTENCE_SET = "persistence_set"


@dataclass
class EndpointEvent:
    """Standardized normalized host telemetry event."""
    event_id:           str = field(default_factory=lambda: f"EPE-{uuid.uuid4().hex[:12]}")
    event_type:         EndpointEventType = EndpointEventType.PROCESS_SPAWN
    timestamp:          float = field(default_factory=time.time)
    host_id:            str = "host-01"
    hostname:           str = "workstation-01.corp.internal"
    user_id:            str = "user-alice"
    
    # Process attributes
    pid:                int = 1000
    ppid:               int = 1
    exe:                str = "/usr/bin/bash"
    cmdline:            str = "/usr/bin/bash -i"
    parent_exe:         str = "/usr/lib/systemd/systemd"
    parent_cmdline:     str = "/sbin/init"
    is_elevated:        bool = False
    is_root:            bool = False
    
    # File attributes
    file_path:          str = ""
    file_operation:     str = ""      # "CREATE", "WRITE", "RENAME", "DELETE"
    file_entropy:       float = 0.0   # Shannon entropy in [0.0, 8.0]
    file_size_bytes:    int = 0
    file_extension:     str = ""
    target_path:        str = ""      # For renames / links
    
    # Network attributes
    src_ip:             str = "10.0.0.10"
    dst_ip:             str = ""
    dst_port:           int = 0
    protocol:           str = "TCP"
    bytes_sent:         int = 0
    bytes_recv:         int = 0
    
    # Auth & Privilege attributes
    auth_success:       bool = True
    auth_failures:      int = 0
    privilege_tier:     str = "USER"  # "USER", "SUDOER", "ROOT", "SYSTEM"
    
    # Persistence attributes
    persistence_type:   str = ""      # "CRON", "SYSTEMD", "REGISTRY_RUN", "BASHRC"
    
    # Raw payload hash & metadata
    raw_hash:           str = ""
    metadata:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class EndpointTelemetryBuffer:
    """
    In-memory rolling buffer for endpoint events per host.
    Supports temporal aggregation for behavioral burst and fan-out detection.
    """

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: List[EndpointEvent] = []

    def append(self, event: EndpointEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events.pop(0)

    def get_recent(self, host_id: Optional[str] = None, window_sec: float = 60.0) -> List[EndpointEvent]:
        now = time.time()
        cutoff = now - window_sec
        return [
            e for e in self._events
            if e.timestamp >= cutoff and (host_id is None or e.host_id == host_id)
        ]

    def clear(self) -> None:
        self._events.clear()
