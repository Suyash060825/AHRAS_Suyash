from __future__ import annotations
"""
AHRAS Peer Group Statistical Engine
----------------------------------
Compares entity behavior against its peer group's baseline distribution.

Attack Pattern Caught:
  Catches insider threats, compromised service accounts, and lateral movement
  where an infected or rogue entity behaves within its own historical limits
  (so single-entity Z-score / EWMA does not fire), but diverges significantly
  from its peer cohort (e.g., a workstation sending 50GB while peer workstations
  average 100MB).

Numerical Algorithm — Welford's Algorithm:
  Uses Welford's online algorithm (Welford 1962) to maintain numerically stable
  rolling mean and sample variance across arbitrary stream lengths without storing
  raw observation vectors or recomputing from scratch.

Peer Group Derivation Rules:
  - network_activity / network_conn: Subnet (/24 IPv4 prefix, e.g. "192.168.1.0/24")
  - process_activity / file_activity: Host prefix (e.g., "web-prod" from "web-prod-01")
  - cloud_api: User role category ("service_account" vs "human_user")
"""

import math
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested dict accessor."""
    curr = d
    for k in keys:
        if not isinstance(curr, dict):
            return default
        curr = curr.get(k)
        if curr is None:
            return default
    return curr


# ─────────────────────────────────────────────────────────────────────────────
# Welford's Online Accumulator
# ─────────────────────────────────────────────────────────────────────────────

class _WelfordAccumulator:
    """
    Numerically stable online computation of mean and variance using Welford's algorithm.
    Allows continuous O(1) updates without unbounded array growth.
    """
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class PeerGroupResult:
    """Output from scoring an entity against its peer group."""
    peer_group_key:     str
    entity_key:         str
    metric_value:       float
    peer_mean:          float
    peer_std:           float
    peer_zscore:        float
    is_peer_anomaly:    bool
    confidence:         float
    flags:              list[str] = field(default_factory=list)
    peer_sample_count:  int = 0
    mitre_techniques:   list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Peer Group Engine
# ─────────────────────────────────────────────────────────────────────────────

class PeerGroupEngine:
    """
    Tracks and scores entity activity relative to peer cohort distributions.
    Thread-safe via RLock.
    """

    def __init__(self, zscore_threshold: float = 3.0, min_peer_samples: int = 10):
        self._groups: Dict[tuple, _WelfordAccumulator] = defaultdict(_WelfordAccumulator)
        self._lock = threading.RLock()
        self._zscore_threshold = zscore_threshold
        self._min_peer_samples = min_peer_samples

    def derive_peer_group_key(self, evt: dict) -> str:
        """Derive peer cohort key from event metadata."""
        cls = _get(evt, "ocsf_class", default="")

        if cls in ("network_activity", "network_conn"):
            ip = _get(evt, "src_endpoint", "ip") or _get(evt, "dst_endpoint", "ip") or ""
            parts = ip.split(".")
            if len(parts) == 4:
                return f"subnet:{parts[0]}.{parts[1]}.{parts[2]}.0/24"
            return "subnet:global"

        if cls in ("process_activity", "file_activity"):
            hostname = _get(evt, "device", "hostname", default="")
            if "-" in hostname:
                prefix = "-".join(hostname.split("-")[:-1])
                return f"hostgroup:{prefix}"
            return f"hostgroup:{hostname}" if hostname else "hostgroup:default"

        if cls == "cloud_api":
            user = _get(evt, "actor", "user", "name", default="")
            if "service" in user.lower() or "sa-" in user.lower() or "bot" in user.lower():
                return "cloudgroup:service_account"
            return "cloudgroup:human_user"

        return "peergroup:general"

    def update_and_score(self, evt: dict, entity_key: str, metric_val: float) -> PeerGroupResult:
        """
        Updates peer group distribution and evaluates entity deviation against peers.
        """
        cls = _get(evt, "ocsf_class", default="other")
        peer_key = self.derive_peer_group_key(evt)

        with self._lock:
            acc = self._groups[(cls, peer_key)]
            
            # Score against current peer baseline BEFORE updating to avoid self-bias
            current_n = acc.n
            current_mean = acc.mean
            current_std = acc.std
            
            # Update peer cohort baseline
            acc.update(metric_val)

        if current_n < self._min_peer_samples:
            return PeerGroupResult(
                peer_group_key=peer_key, entity_key=entity_key, metric_value=metric_val,
                peer_mean=current_mean, peer_std=current_std, peer_zscore=0.0,
                is_peer_anomaly=False, confidence=0.0, peer_sample_count=current_n
            )

        eff_std = max(current_std, abs(current_mean) * 0.05, 1e-3)
        peer_z = abs(metric_val - current_mean) / eff_std

        flags = []
        mitre = []
        is_anomaly = False

        if peer_z > self._zscore_threshold:
            flags.append(f"peer_zscore={peer_z:.2f} (group={peer_key})")
            mitre.append("T1078")  # Valid Accounts: Peer anomaly
            is_anomaly = True

        confidence = min(peer_z / (self._zscore_threshold * 2), 1.0)
        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)

        return PeerGroupResult(
            peer_group_key=peer_key,
            entity_key=entity_key,
            metric_value=round(metric_val, 4),
            peer_mean=round(current_mean, 4),
            peer_std=round(current_std, 4),
            peer_zscore=round(peer_z, 3),
            is_peer_anomaly=is_anomaly,
            confidence=confidence,
            flags=flags,
            peer_sample_count=current_n,
            mitre_techniques=mitre,
        )


# Singleton
_peer_engine_instance: Optional[PeerGroupEngine] = None
_peer_lock = threading.Lock()


def get_peer_group_engine() -> PeerGroupEngine:
    global _peer_engine_instance
    with _peer_lock:
        if _peer_engine_instance is None:
            _peer_engine_instance = PeerGroupEngine()
    return _peer_engine_instance
