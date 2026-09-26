from __future__ import annotations
"""
AHRAS Module — Encrypted Session Intelligence (Extension G)
============================================================
Infers malicious intent from encrypted communication sessions (TLS 1.3, QUIC, SSH)
WITHOUT decrypting payloads, leveraging packet size dynamics, direction transitions,
inter-arrival time (IAT) periodicity, burst structures, and protocol metadata.

Directly resolves Botnet / C2 Beaconing stealth failures where payload inspection
is impossible and aggregate flow volume looks statistically normal.

Representations:
  1. Short Sequence Representation: P in R^(L x 3) [size, direction, log_iat]
  2. Fixed-Size Session Vector: v_session in R^24 [moments, periodicity, burstiness]

Emits:
  Standardized EvidenceRecord for risk engine and decision trace integration.
"""

import math
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    USE_ENCRYPTED_SESSION_INTEL,
    SESSION_SEQUENCE_LEN,
    BEACON_PERIODICITY_THRESHOLD,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts & Schemas
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PacketMetadata:
    """Metadata for an individual encrypted packet (payload remains untouched)."""
    size: int                         # Wire size in bytes (e.g. 54 to 1514)
    direction: int                    # +1 = Client -> Server (Outbound), -1 = Server -> Client (Inbound)
    timestamp: float                  # Epoch timestamp in seconds
    tcp_flags: List[str] = field(default_factory=list)


@dataclass
class SessionMetadata:
    """Session-level protocol metadata."""
    alpn: str = "h2"                  # h2, http/1.1, spdy, quic
    sni: str = "api.enterprise.net"   # Server Name Indication
    cipher_suite: str = "TLS_AES_256_GCM_SHA384"
    tls_version: str = "TLSv1.3"
    extension_count: int = 12
    resumed_session: bool = False


@dataclass
class SessionEvidenceRecord:
    """Standardized EvidenceRecord emitted by the Encrypted Session Intelligence engine."""
    evidence_id: str
    event_id: str
    entity_id: str
    source: str = "encrypted_session"
    detector_type: str = "encrypted_session_intel"
    threat_label: str = "BENIGN"      # BENIGN | C2_BEACONING | DATA_EXFILTRATION | INTERACTIVE_SHELL
    raw_score: float = 0.0            # Continuous threat anomaly score [0, inf)
    normalized_score: float = 0.0     # Calibrated score [0.0, 1.0]
    confidence: float = 0.90          # Confidence in verdict [0.0, 1.0]
    uncertainty: float = 0.10         # Epistemic uncertainty [0.0, 1.0]
    mitre_technique: Optional[str] = None # e.g. T1071.001 (Web Protocols), T1573 (Encrypted Channel)
    beacon_periodicity: float = 0.0   # Autocorrelation peak in [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction & Periodicity Analytics
# ─────────────────────────────────────────────────────────────────────────────

def compute_autocorrelation(series: np.ndarray, max_lag: int = 10) -> float:
    """
    Computes beaconing periodicity of inter-arrival times.
    Combines normalized lag autocorrelation with low coefficient of variation.
    High score (rho >= 0.75) indicates regular programmatic C2 beaconing.
    """
    if len(series) < 4:
        return 0.0

    mean_val = float(np.mean(series))
    std_val = float(np.std(series))
    if mean_val < 1e-6:
        return 1.0  # Constant zero intervals

    cv = std_val / mean_val
    cv_score = max(0.0, 1.0 - min(1.0, cv))

    # Also compute normalized lag autocorrelation across lags 1..max_lag
    centered = series - mean_val
    denom = float(np.sum(centered ** 2))
    max_rho = 0.0
    if denom > 1e-12:
        k_max = min(max_lag, len(series) // 2)
        for k in range(1, max(2, k_max)):
            nom = float(np.sum(centered[:-k] * centered[k:]))
            rho = nom / denom
            if rho > max_rho:
                max_rho = rho

    # Periodicity is high if either lag autocorrelation is strong OR variance is extremely low
    combined = max(cv_score, max_rho)
    return float(np.clip(combined, 0.0, 1.0))


def compute_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy of string identifiers (SNI, domain, JA4)."""
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    ent = 0.0
    n = len(text)
    for cnt in counts.values():
        p = cnt / n
        ent -= p * math.log2(p)
    return float(ent)


# ─────────────────────────────────────────────────────────────────────────────
# Encrypted Session Intelligence Engine
# ─────────────────────────────────────────────────────────────────────────────

class EncryptedSessionIntelligence:
    """
    Extracts sequence dynamics and statistical moments from encrypted sessions
    and classifies traffic without inspecting or decrypting payloads.
    """

    def __init__(
        self,
        sequence_len: int = SESSION_SEQUENCE_LEN,
        periodicity_threshold: float = BEACON_PERIODICITY_THRESHOLD,
        enabled: bool = USE_ENCRYPTED_SESSION_INTEL,
    ):
        self.sequence_len = sequence_len
        self.periodicity_threshold = periodicity_threshold
        self.enabled = enabled
        self._lock = threading.Lock()

    # ── Representation 1: Short Sequence Representation ──────────────────────

    def extract_sequence(
        self, packets: List[PacketMetadata]
    ) -> np.ndarray:
        """
        Creates a short sequence representation matrix P in R^(L x 3).
        Each row: [normalized_size, direction, log_inter_arrival_time].
        """
        L = self.sequence_len
        seq = np.zeros((L, 3), dtype=np.float32)
        n = min(len(packets), L)
        if n == 0:
            return seq

        for i in range(n):
            pkt = packets[i]
            # 1. Size normalized to MTU 1500
            norm_size = min(1.0, max(0.0, pkt.size / 1500.0))
            # 2. Direction: +1.0 or -1.0
            direction = 1.0 if pkt.direction >= 0 else -1.0
            # 3. Inter-arrival time (log scale)
            if i == 0:
                iat = 0.0
            else:
                iat = max(0.0, pkt.timestamp - packets[i - 1].timestamp)
            log_iat = math.log1p(iat)

            seq[i] = [norm_size, direction, log_iat]

        return seq

    # ── Representation 2: Fixed-Size Session Vector ──────────────────────────

    def extract_session_vector(
        self,
        packets: List[PacketMetadata],
        meta: Optional[SessionMetadata] = None,
    ) -> np.ndarray:
        """
        Constructs a comprehensive 24-dimensional session feature vector.
        Moments, directional transitions, IAT autocorrelation, and protocol metadata.
        """
        meta = meta or SessionMetadata()
        if not packets:
            return np.zeros(24, dtype=np.float32)

        sizes = np.array([p.size for p in packets], dtype=np.float64)
        dirs = np.array([p.direction for p in packets], dtype=np.float64)
        ts = np.array([p.timestamp for p in packets], dtype=np.float64)

        # IATs
        iats = np.diff(ts) if len(ts) > 1 else np.array([0.0])
        iats = np.maximum(0.0, iats)

        # 1. Size moments (6 dims)
        mean_size = float(np.mean(sizes))
        std_size = float(np.std(sizes))
        min_size = float(np.min(sizes))
        max_size = float(np.max(sizes))
        size_skew = float(np.mean(((sizes - mean_size) / (std_size + 1e-6)) ** 3))
        size_entropy = compute_shannon_entropy("".join(str(int(s % 10)) for s in sizes[:50]))

        # 2. Direction dynamics (4 dims)
        out_ratio = float(np.mean(dirs > 0))
        in_ratio = 1.0 - out_ratio
        dir_flips = float(np.sum(dirs[:-1] != dirs[1:])) if len(dirs) > 1 else 0.0
        flip_rate = dir_flips / max(1, len(dirs) - 1)

        # 3. Timing & Periodicity (5 dims)
        mean_iat = float(np.mean(iats))
        std_iat = float(np.std(iats))
        cv_iat = std_iat / (mean_iat + 1e-6)  # Coefficient of variation (low in beacons)
        beacon_score = compute_autocorrelation(iats)
        duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

        # 4. Burst structure (4 dims)
        # Contiguous packets in same direction
        bursts = []
        cur_len = 1
        for i in range(1, len(dirs)):
            if dirs[i] == dirs[i - 1]:
                cur_len += 1
            else:
                bursts.append(cur_len)
                cur_len = 1
        bursts.append(cur_len)
        mean_burst = float(np.mean(bursts))
        max_burst = float(np.max(bursts))
        burst_count = float(len(bursts))
        burst_rate = burst_count / max(0.1, duration + 0.1)

        # 5. Metadata features (5 dims)
        sni_entropy = compute_shannon_entropy(meta.sni)
        alpn_val = 1.0 if meta.alpn == "h2" else (2.0 if meta.alpn == "quic" else 0.5)
        ext_count = float(meta.extension_count)
        resumed = 1.0 if meta.resumed_session else 0.0
        total_bytes = float(np.sum(sizes))

        vec = np.array([
            mean_size / 1500.0,
            std_size / 1500.0,
            min_size / 1500.0,
            max_size / 1500.0,
            math.tanh(size_skew),
            size_entropy / 8.0,
            out_ratio,
            in_ratio,
            flip_rate,
            min(10.0, dir_flips) / 10.0,
            math.log1p(mean_iat),
            math.tanh(cv_iat),
            beacon_score,
            math.log1p(duration),
            math.log1p(len(packets)),
            mean_burst / 20.0,
            max_burst / 50.0,
            math.log1p(burst_count),
            math.tanh(burst_rate),
            sni_entropy / 8.0,
            alpn_val / 2.0,
            ext_count / 20.0,
            resumed,
            math.log1p(total_bytes) / 15.0,
        ], dtype=np.float32)

        return vec

    # ── Detector & Threat Classification ─────────────────────────────────────

    def analyze_session(
        self,
        event_id: str,
        entity_id: str,
        packets: List[PacketMetadata],
        meta: Optional[SessionMetadata] = None,
    ) -> SessionEvidenceRecord:
        """
        Analyzes encrypted session telemetry and produces a standardized EvidenceRecord.
        """
        meta = meta or SessionMetadata()
        t0 = time.perf_counter()

        if not packets:
            return SessionEvidenceRecord(
                evidence_id=str(uuid.uuid4()),
                event_id=event_id,
                entity_id=entity_id,
                threat_label="BENIGN",
                raw_score=0.0,
                normalized_score=0.0,
                confidence=0.50,
                uncertainty=0.50,
                mitre_technique=None,
                metadata={"reason": "empty_session"},
            )

        seq = self.extract_sequence(packets)
        vec = self.extract_session_vector(packets, meta)

        # Extracted key diagnostic metrics
        beacon_score = float(vec[12])      # Autocorrelation peak
        cv_iat = float(vec[11])            # IAT coefficient of variation
        mean_size = float(vec[0] * 1500.0) # Mean size
        out_ratio = float(vec[6])          # Outbound volume ratio
        duration = float(math.expm1(vec[13]))
        flip_rate = float(vec[8])

        sizes_arr = np.array([p.size for p in packets], dtype=np.float64)
        dirs_arr = np.array([p.direction for p in packets], dtype=np.float64)
        bytes_out = float(np.sum(sizes_arr[dirs_arr > 0]))
        bytes_total = float(np.sum(sizes_arr))
        byte_out_ratio = bytes_out / max(1.0, bytes_total)

        threat_label = "BENIGN"
        mitre_technique = None
        raw_score = 0.0
        normalized_score = 0.0
        confidence = 0.85
        uncertainty = 0.15

        # Heuristic 1: Stealth C2 Beaconing (Low volume, high timing periodicity, small payloads)
        # Resolves Botnet F1=0.0 by catching regular heartbeat intervals without inspecting content
        is_beacon = (
            beacon_score >= self.periodicity_threshold
            and len(packets) >= 8
            and mean_size <= 450.0
        )
        if is_beacon:
            threat_label = "C2_BEACONING"
            mitre_technique = "T1071.001"  # Web Protocols C2
            raw_score = 2.5 + beacon_score * 2.0
            normalized_score = float(np.clip(0.75 + 0.24 * beacon_score, 0.75, 0.99))
            confidence = 0.94
            uncertainty = round(0.20 * (1.0 - beacon_score), 4)

        # Heuristic 2: Encrypted Data Exfiltration (Heavy outbound byte bias, large packet sizes, low return traffic)
        elif (out_ratio >= 0.80 or byte_out_ratio >= 0.88) and mean_size >= 900.0 and len(packets) >= 15:
            threat_label = "DATA_EXFILTRATION"
            mitre_technique = "T1041"      # Exfiltration Over C2 Channel
            raw_score = 3.0 * byte_out_ratio
            normalized_score = float(np.clip(0.80 + 0.18 * byte_out_ratio, 0.80, 0.98))
            confidence = 0.92
            uncertainty = 0.12

        # Heuristic 3: Interactive Reverse Shell (Human typing cadence, keystroke packet sizes, high bidirectional flips)
        elif (
            0.35 <= out_ratio <= 0.65
            and 60.0 <= mean_size <= 220.0
            and flip_rate >= 0.40
            and len(packets) >= 12
            and beacon_score < 0.70  # Human keystrokes are aperiodic
        ):
            threat_label = "INTERACTIVE_SHELL"
            mitre_technique = "T1059"      # Command and Scripting Interpreter
            raw_score = 1.8 + flip_rate
            normalized_score = float(np.clip(0.68 + 0.22 * flip_rate, 0.65, 0.90))
            confidence = 0.86
            uncertainty = 0.20

        # Normal benign encrypted traffic (HTTPS browsing, streaming)
        else:
            threat_label = "BENIGN"
            mitre_technique = None
            raw_score = 0.10
            normalized_score = 0.05
            confidence = 0.90
            uncertainty = 0.10

        proc_ms = round((time.perf_counter() - t0) * 1000, 3)

        return SessionEvidenceRecord(
            evidence_id=str(uuid.uuid4()),
            event_id=event_id,
            entity_id=entity_id,
            threat_label=threat_label,
            raw_score=round(raw_score, 4),
            normalized_score=round(normalized_score, 4),
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            mitre_technique=mitre_technique,
            beacon_periodicity=round(beacon_score, 4),
            metadata={
                "mean_packet_size": round(mean_size, 1),
                "outbound_ratio": round(out_ratio, 3),
                "flip_rate": round(flip_rate, 3),
                "duration_seconds": round(duration, 2),
                "packet_count": len(packets),
                "processing_ms": proc_ms,
                "session_vector": vec.tolist(),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module Singleton
# ─────────────────────────────────────────────────────────────────────────────

_session_intel_instance: Optional[EncryptedSessionIntelligence] = None
_session_intel_lock = threading.Lock()


def get_encrypted_session_intel() -> EncryptedSessionIntelligence:
    """Returns thread-safe singleton instance of EncryptedSessionIntelligence."""
    global _session_intel_instance
    with _session_intel_lock:
        if _session_intel_instance is None:
            _session_intel_instance = EncryptedSessionIntelligence()
    return _session_intel_instance
