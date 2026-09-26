from __future__ import annotations
"""
AHRAS Module — Streaming Sketch Fast Path (Extension F)
======================================================
Memory-efficient O(1) first-line telemetry summarizer and heavy-hitter screener
for high-throughput streaming environments (>100k events/sec).

Provides:
  1. Count-Min Sketches:
     - Source entity event frequency
     - Destination entity event frequency
     - Packet volume accumulation
     - Byte volume accumulation
     - Distinct 5-tuple flow tracking
  2. Cardinality Estimation (HyperLogLog registers):
     - Source fan-out (distinct destination IP count per source)
  3. Temporal Burstiness & Velocity:
     - Dual-window exponential decay delta
  4. Heavy-Hitter Screening:
     - Volume heavy hitters (volumetric DDoS, brute force)
     - High fan-out scanners (horizontal network scanning)
     - Destination burst targets (SYN/UDP flood targets)
  5. Mathematical Invariant & Bounded Error:
     - Theoretical guarantee: a_i <= hat{a}_i <= a_i + epsilon * ||a||_1
       with probability >= 1 - delta, where epsilon = e / width, delta = e^(-depth).
     - Strictly used for FAST SCREENING and FEATURE SUMMARIZATION.
     - Never substitutes approximate values when exact forensic values are available.
"""

import math
import time
import uuid
import struct
import hashlib
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config.settings import (
    USE_STREAMING_SKETCH,
    SKETCH_WIDTH,
    SKETCH_DEPTH,
    SKETCH_HEAVY_HITTER_THRESHOLD,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sketch Configuration & Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SketchConfig:
    width: int = SKETCH_WIDTH                    # w = 4096 -> epsilon = e/4096 = 0.000664
    depth: int = SKETCH_DEPTH                    # d = 5 -> delta = e^(-5) = 0.0067 (99.33% confidence)
    heavy_hitter_threshold: float = SKETCH_HEAVY_HITTER_THRESHOLD # phi = 0.01 (1% of stream)
    fanout_threshold: int = 15                  # Distinct destinations considered horizontal scan
    burst_multiplier: float = 3.0               # Window-over-window burst threshold
    window_seconds: float = 10.0                # Active window duration before decay
    hll_registers: int = 32                     # HyperLogLog register count per source bucket (std error 18%)
    enabled: bool = USE_STREAMING_SKETCH


@dataclass
class SketchEvidenceRecord:
    """
    Standardized EvidenceRecord emitted when the streaming sketch detects a
    heavy-hitter, fan-out scanner, or destination burst anomaly.
    """
    evidence_id: str
    event_id: str
    entity_id: str
    source: str = "sketch"
    detector_type: str = "streaming_sketch"
    anomaly_type: str = "heavy_hitter_volume"  # heavy_hitter_volume | high_fanout_scan | destination_burst
    estimated_value: float = 0.0
    error_bound: float = 0.0                   # Theoretical epsilon * N
    exact_value_available: bool = False        # Sketch values are approximate by design
    confidence: float = 0.90
    uncertainty: float = 0.10
    mitre_technique: Optional[str] = None      # e.g., T1046 (Network Service Discovery), T1498 (Network DoS)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SketchScreenResult:
    """Fast-screening verdict emitted for line-rate event triage."""
    event_id: str
    is_flagged: bool
    reasons: list[str] = field(default_factory=list)
    evidence_records: list[SketchEvidenceRecord] = field(default_factory=list)
    src_est_count: int = 0
    dst_est_count: int = 0
    src_fanout_est: int = 0
    burst_ratio: float = 1.0
    processing_us: float = 0.0  # Microseconds


# ─────────────────────────────────────────────────────────────────────────────
# Universal Hash Functions
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pair(key: str, seed: int) -> Tuple[int, int]:
    """Generates 64-bit hash components using blake2b for 2-universal hash family."""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=seed.to_bytes(4, "big")).digest()
    v = struct.unpack(">Q", h)[0]
    return (v >> 32) & 0xFFFFFFFF, v & 0xFFFFFFFF


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Sketch Fast-Path Engine
# ─────────────────────────────────────────────────────────────────────────────

class StreamingSketchEngine:
    """
    Thread-safe, O(1) space and O(1) time telemetry summarizer and anomaly screener.
    Maintains fixed-size Count-Min Sketches and HyperLogLog cardinality structures.
    """

    def __init__(self, config: Optional[SketchConfig] = None):
        self.config = config or SketchConfig()
        self._lock = threading.Lock()

        w = self.config.width
        d = self.config.depth

        # Theoretical approximation guarantees
        self.epsilon = math.e / w
        self.delta = math.exp(-d)

        # Count-Min Sketch tables (w x d)
        self._src_cm = np.zeros((d, w), dtype=np.uint32)
        self._dst_cm = np.zeros((d, w), dtype=np.uint32)
        self._pkt_cm = np.zeros((d, w), dtype=np.uint64)
        self._byte_cm = np.zeros((d, w), dtype=np.uint64)
        self._flow_cm = np.zeros((d, w), dtype=np.uint32)

        # Sliding burst tracking: current active counts vs previous window snapshot
        self._cur_window_counts = np.zeros((d, w), dtype=np.uint32)
        self._prev_window_counts = np.zeros((d, w), dtype=np.uint32)
        self._window_start = time.time()

        # HyperLogLog registers for fan-out (distinct destination IP cardinality per source)
        # Using a bounded 2D register array of (w, hll_registers)
        m = self.config.hll_registers
        self._fanout_hll = np.zeros((w, m), dtype=np.uint8)

        # Stream counters
        self.total_events = 0
        self.total_packets = 0
        self.total_bytes = 0

        # Heavy-Hitter bounded cache: (entity_id -> (timestamp, est_count))
        # Strictly bounded to at most 512 entries to maintain strict O(1) space
        self._heavy_hitters_cache: Dict[str, Tuple[float, int]] = {}

    def _get_indices(self, key: str) -> List[int]:
        """Calculates row bucket indices across d hash functions."""
        indices = []
        w = self.config.width
        for i in range(self.config.depth):
            a, b = _hash_pair(key, seed=i * 10007 + 3)
            idx = (a ^ b) % w
            indices.append(idx)
        return indices

    def _update_hll(self, src_key: str, dst_key: str) -> None:
        """Updates HyperLogLog registers for source entity fan-out estimation."""
        w = self.config.width
        m = self.config.hll_registers
        src_bucket = _hash_pair(src_key, seed=99991)[0] % w

        # Hash dst_key to get register index and leading zeros
        dst_h = _hash_pair(dst_key, seed=77773)[0]
        reg_idx = dst_h % m
        # Count leading zeros in remainder + 1
        rem = (dst_h // m) | 1
        leading_zeros = (rem.bit_length() ^ 32) if rem < (1 << 31) else 1
        rho = min(32, max(1, 32 - rem.bit_length() + 1))

        if rho > self._fanout_hll[src_bucket, reg_idx]:
            self._fanout_hll[src_bucket, reg_idx] = rho

    def _estimate_hll(self, src_key: str) -> int:
        """Estimates distinct cardinality using raw HyperLogLog estimator."""
        w = self.config.width
        m = self.config.hll_registers
        src_bucket = _hash_pair(src_key, seed=99991)[0] % w

        regs = self._fanout_hll[src_bucket]
        # Raw indicator harmonic sum: sum(2^(-M[j]))
        indicator = float(np.sum(2.0 ** (-regs.astype(np.float64))))
        if indicator == 0:
            return 0

        # Alpha bias correction for m=32 or m=16
        if m == 16:
            alpha = 0.673
        elif m == 32:
            alpha = 0.697
        elif m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / m)

        raw_est = alpha * (m ** 2) / indicator

        # Small range linear counting correction
        if raw_est <= 2.5 * m:
            v = int(np.sum(regs == 0))
            if v > 0:
                raw_est = m * math.log(m / v)

        return max(1, int(round(raw_est)))

    def _check_and_rotate_window(self, now: float) -> None:
        """Rotates sliding window for burstiness quantification."""
        if now - self._window_start >= self.config.window_seconds:
            self._prev_window_counts = self._cur_window_counts.copy()
            self._cur_window_counts.fill(0)
            self._window_start = now
            # Prune stale heavy-hitters cache (> 60s)
            stale_keys = [k for k, (t, _) in self._heavy_hitters_cache.items() if now - t > 60.0]
            for k in stale_keys:
                del self._heavy_hitters_cache[k]

    # ── Core Ingestion & Update ──────────────────────────────────────────────

    def update(self, evt: Dict[str, Any]) -> None:
        """
        Updates streaming sketches with event metrics in O(1) time.
        """
        now = time.time()
        src_ip = str(evt.get("src_ip", evt.get("source_ip", "0.0.0.0")))
        dst_ip = str(evt.get("dst_ip", evt.get("destination_ip", "0.0.0.0")))
        src_port = evt.get("src_port", 0)
        dst_port = evt.get("dst_port", 0)
        proto = str(evt.get("protocol", "TCP"))

        pkts = max(1, int(evt.get("packet_count", evt.get("pkts_in", 1))))
        bytes_cnt = max(0, int(evt.get("byte_count", evt.get("bytes_in", 0)) + evt.get("bytes_out", 0)))
        flow_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}/{proto}"

        src_indices = self._get_indices(src_ip)
        dst_indices = self._get_indices(dst_ip)
        flow_indices = self._get_indices(flow_key)

        with self._lock:
            self._check_and_rotate_window(now)
            self.total_events += 1
            self.total_packets += pkts
            self.total_bytes += bytes_cnt

            for row, (s_idx, d_idx, f_idx) in enumerate(zip(src_indices, dst_indices, flow_indices)):
                self._src_cm[row, s_idx] += 1
                self._dst_cm[row, d_idx] += 1
                self._pkt_cm[row, s_idx] += pkts
                self._byte_cm[row, s_idx] += bytes_cnt
                self._flow_cm[row, f_idx] += 1
                self._cur_window_counts[row, d_idx] += 1

            self._update_hll(src_ip, dst_ip)

    # ── Query & Estimation ───────────────────────────────────────────────────

    def query_source_frequency(self, src_ip: str) -> Tuple[int, float]:
        """
        Queries source count estimate.
        Returns: (estimated_count, theoretical_error_bound).
        """
        indices = self._get_indices(src_ip)
        with self._lock:
            vals = [self._src_cm[row, idx] for row, idx in enumerate(indices)]
            est = int(min(vals))
            err_bound = round(float(self.epsilon * self.total_events), 2)
            return est, err_bound

    def query_destination_frequency(self, dst_ip: str) -> Tuple[int, float]:
        """
        Queries destination count estimate.
        Returns: (estimated_count, theoretical_error_bound).
        """
        indices = self._get_indices(dst_ip)
        with self._lock:
            vals = [self._dst_cm[row, idx] for row, idx in enumerate(indices)]
            est = int(min(vals))
            err_bound = round(float(self.epsilon * self.total_events), 2)
            return est, err_bound

    def query_source_volume(self, src_ip: str) -> Tuple[int, int]:
        """Returns (estimated_packet_count, estimated_byte_count)."""
        indices = self._get_indices(src_ip)
        with self._lock:
            pkts = int(min([self._pkt_cm[row, idx] for row, idx in enumerate(indices)]))
            bytes_cnt = int(min([self._byte_cm[row, idx] for row, idx in enumerate(indices)]))
            return pkts, bytes_cnt

    def query_source_fanout(self, src_ip: str) -> int:
        """Returns approximate count of distinct destinations contacted by source."""
        with self._lock:
            return self._estimate_hll(src_ip)

    def query_destination_burst(self, dst_ip: str) -> float:
        """Calculates window-over-window traffic burst ratio for a destination."""
        indices = self._get_indices(dst_ip)
        with self._lock:
            cur_vals = [self._cur_window_counts[row, idx] for row, idx in enumerate(indices)]
            prev_vals = [self._prev_window_counts[row, idx] for row, idx in enumerate(indices)]
            cur_cnt = float(min(cur_vals))
            prev_cnt = float(min(prev_vals))
            if prev_cnt <= 5.0:
                return 1.0 if cur_cnt <= 15.0 else round(cur_cnt / 5.0, 2)
            return round(cur_cnt / prev_cnt, 2)

    # ── Fast Screening Triage ────────────────────────────────────────────────

    def screen(self, evt: Dict[str, Any]) -> SketchScreenResult:
        """
        Executes sub-millisecond screening triage on one incoming telemetry event.
        Detects volume heavy hitters, fan-out scanners, and destination bursts.
        """
        t0 = time.perf_counter()
        event_id = str(evt.get("event_id", uuid.uuid4()))
        src_ip = str(evt.get("src_ip", evt.get("source_ip", "0.0.0.0")))
        dst_ip = str(evt.get("dst_ip", evt.get("destination_ip", "0.0.0.0")))

        # 1. Update sketch states
        self.update(evt)

        # 2. Query estimates
        src_cnt, err_bound = self.query_source_frequency(src_ip)
        dst_cnt, _ = self.query_destination_frequency(dst_ip)
        fanout = self.query_source_fanout(src_ip)
        burst_ratio = self.query_destination_burst(dst_ip)

        reasons = []
        evidence_records: List[SketchEvidenceRecord] = []
        is_flagged = False

        # Thresholds
        hh_threshold = max(20, int(self.config.heavy_hitter_threshold * self.total_events))

        # Check 1: Volume Heavy Hitter (DDoS / Volumetric Scan / Brute Force)
        if src_cnt >= hh_threshold:
            is_flagged = True
            reasons.append(f"src_heavy_hitter (est={src_cnt} >= thresh={hh_threshold})")
            evidence_records.append(
                SketchEvidenceRecord(
                    evidence_id=str(uuid.uuid4()),
                    event_id=event_id,
                    entity_id=src_ip,
                    anomaly_type="heavy_hitter_volume",
                    estimated_value=float(src_cnt),
                    error_bound=err_bound,
                    confidence=round(min(0.95, 0.70 + (src_cnt / (hh_threshold * 2 + 1)) * 0.25), 3),
                    uncertainty=round(self.delta, 4),
                    mitre_technique="T1498",
                    metadata={
                        "threshold": hh_threshold,
                        "total_stream_events": self.total_events,
                        "approximation_bound": f"hat_a <= a + {self.epsilon:.6f}*N",
                    },
                )
            )

        # Check 2: High Fan-Out Scanner (Horizontal Port Scan / Reconnaissance)
        if fanout >= self.config.fanout_threshold:
            is_flagged = True
            reasons.append(f"high_fanout_scan (est_distinct_destinations={fanout} >= thresh={self.config.fanout_threshold})")
            evidence_records.append(
                SketchEvidenceRecord(
                    evidence_id=str(uuid.uuid4()),
                    event_id=event_id,
                    entity_id=src_ip,
                    anomaly_type="high_fanout_scan",
                    estimated_value=float(fanout),
                    error_bound=round(fanout * 0.18, 2),  # HLL 18% standard error
                    confidence=0.88,
                    uncertainty=0.18,
                    mitre_technique="T1046",
                    metadata={
                        "distinct_destinations_est": fanout,
                        "cardinality_estimator": "HyperLogLog(m=32)",
                    },
                )
            )

        # Check 3: Destination Burst (Targeted DoS / Service Exhaustion)
        if burst_ratio >= self.config.burst_multiplier and dst_cnt >= 30:
            is_flagged = True
            reasons.append(f"destination_burst (burst_ratio={burst_ratio:.1f}x >= thresh={self.config.burst_multiplier:.1f}x)")
            evidence_records.append(
                SketchEvidenceRecord(
                    evidence_id=str(uuid.uuid4()),
                    event_id=event_id,
                    entity_id=dst_ip,
                    anomaly_type="destination_burst",
                    estimated_value=float(burst_ratio),
                    error_bound=0.20,
                    confidence=0.85,
                    uncertainty=0.15,
                    mitre_technique="T1499",
                    metadata={
                        "burst_ratio": burst_ratio,
                        "dst_total_count": dst_cnt,
                    },
                )
            )

        # Update heavy-hitters cache
        if is_flagged:
            with self._lock:
                if len(self._heavy_hitters_cache) < 512:
                    self._heavy_hitters_cache[src_ip] = (time.time(), src_cnt)

        us = round((time.perf_counter() - t0) * 1_000_000, 2)

        return SketchScreenResult(
            event_id=event_id,
            is_flagged=is_flagged,
            reasons=reasons,
            evidence_records=evidence_records,
            src_est_count=src_cnt,
            dst_est_count=dst_cnt,
            src_fanout_est=fanout,
            burst_ratio=burst_ratio,
            processing_us=us,
        )

    def get_memory_footprint_bytes(self) -> int:
        """Calculates exact memory footprint of all sketch arrays in bytes."""
        total = (
            self._src_cm.nbytes
            + self._dst_cm.nbytes
            + self._pkt_cm.nbytes
            + self._byte_cm.nbytes
            + self._flow_cm.nbytes
            + self._cur_window_counts.nbytes
            + self._prev_window_counts.nbytes
            + self._fanout_hll.nbytes
        )
        return total

    def get_stats(self) -> Dict[str, Any]:
        """Returns operational summary and approximation bounds."""
        with self._lock:
            mem_kb = round(self.get_memory_footprint_bytes() / 1024, 2)
            return {
                "total_events": self.total_events,
                "total_packets": self.total_packets,
                "total_bytes": self.total_bytes,
                "memory_footprint_kb": mem_kb,
                "memory_footprint_mb": round(mem_kb / 1024, 3),
                "width": self.config.width,
                "depth": self.config.depth,
                "epsilon": round(self.epsilon, 6),
                "delta": round(self.delta, 6),
                "confidence_pct": round((1.0 - self.delta) * 100, 2),
                "active_heavy_hitters_cached": len(self._heavy_hitters_cache),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_sketch_engine_instance: Optional[StreamingSketchEngine] = None
_sketch_engine_lock = threading.Lock()


def get_sketch_engine(config: Optional[SketchConfig] = None) -> StreamingSketchEngine:
    """Returns thread-safe singleton instance of StreamingSketchEngine."""
    global _sketch_engine_instance
    with _sketch_engine_lock:
        if _sketch_engine_instance is None or config is not None:
            _sketch_engine_instance = StreamingSketchEngine(config=config)
    return _sketch_engine_instance
