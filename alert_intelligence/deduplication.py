from __future__ import annotations
"""
AHRAS Adaptive Alert Deduplication Engine
==========================================
Mitigates alert flooding from repetitive high-frequency detection bursts
(e.g., DoS volume floods, high-rate port scanning, repetitive brute-force attempts).

Operational Invariants:
1. Zero Evidence Loss: All cryptographic evidence_ids from suppressed alerts
   are preserved and merged into the canonical alert instance.
2. Bounded Memory: Old deduplication keys expire after sliding window `window_seconds`.
3. Geometric Notification Thresholds: Updates are emitted on exponential powers
   (10, 50, 100, 500, 1000, ...) to prevent downstream queue exhaustion.
"""

import time
import threading
from typing import Callable, Dict, List, Optional, Tuple
from alert_intelligence.models import RawAlert


class AdaptiveAlertDeduplicator:
    """
    Sliding-window deduplicator that collapses duplicate alert bursts
    while retaining all cryptographic evidence identifiers and updating intensity metrics.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        geometric_multiplier: float = 2.0,
        min_suppress_before_emit: int = 5,
        on_evict: Optional[Callable[[RawAlert], None]] = None
    ):
        self.window_seconds = window_seconds
        self.geometric_multiplier = geometric_multiplier
        self.min_suppress_before_emit = min_suppress_before_emit
        self.on_evict = on_evict
        self._lock = threading.Lock()
        
        # State: dedup_key -> Canonical alert representation
        # Key format: (entity_key, source_engine, detector_name, mitre_technique)
        self._active_window: Dict[Tuple[str, str, str, Optional[str]], RawAlert] = {}
        self._last_emission_count: Dict[Tuple[str, str, str, Optional[str]], int] = {}
        self._total_ingested = 0
        self._total_suppressed = 0

    def _make_key(self, alert: RawAlert) -> Tuple[str, str, str, Optional[str]]:
        return (alert.entity_key, alert.source_engine, alert.detector_name, alert.mitre_technique)

    def process_alert(self, alert: RawAlert) -> Tuple[Optional[RawAlert], bool]:
        """
        Processes an arriving RawAlert.
        
        Returns:
            Tuple[Optional[RawAlert], bool]:
                - (emitted_alert, is_new):
                    - If brand new alert: returns (alert, True)
                    - If suppressed within window: returns (None, False)
                    - If geometric milestone reached after min_suppress: returns (updated_alert, False)
        """
        now = alert.timestamp if alert.timestamp > 0 else time.time()
        key = self._make_key(alert)
        
        with self._lock:
            self._total_ingested += 1
            self._prune_expired(now)
            
            if key not in self._active_window:
                # First time seeing this alert in current window
                canonical = alert.model_copy(deep=True)
                canonical.suppressed_count = 0
                self._active_window[key] = canonical
                self._last_emission_count[key] = 1
                return canonical, True
            
            # Duplicate found within sliding window
            canonical = self._active_window[key]
            canonical.suppressed_count += 1
            self._total_suppressed += 1
            
            # Invariant: Merge all cryptographic evidence references without loss
            for ev_id in alert.evidence_ids:
                if ev_id not in canonical.evidence_ids:
                    canonical.evidence_ids.append(ev_id)
            
            # Update peak severity/score if the duplicate exhibited higher severity
            if alert.normalized_score > canonical.normalized_score:
                canonical.normalized_score = alert.normalized_score
                canonical.raw_score = alert.raw_score
                canonical.confidence = max(canonical.confidence, alert.confidence)
            
            total_count = canonical.suppressed_count + 1
            last_emitted = self._last_emission_count.get(key, 1)
            
            # Check geometric emission milestone (only after min_suppress_before_emit)
            if total_count >= self.min_suppress_before_emit and total_count >= last_emitted * self.geometric_multiplier:
                self._last_emission_count[key] = total_count
                # Emit milestone update
                return canonical.model_copy(deep=True), False
            
            return None, False

    def _prune_expired(self, current_time: float) -> None:
        """Prunes window keys older than window_seconds, invoking eviction callback to prevent data loss."""
        cutoff = current_time - self.window_seconds
        expired_keys = [
            k for k, alert in self._active_window.items()
            if alert.timestamp < cutoff
        ]
        for k in expired_keys:
            canonical = self._active_window.pop(k)
            self._last_emission_count.pop(k, None)
            if self.on_evict and canonical.suppressed_count > 0:
                self.on_evict(canonical)

    def flush(self) -> List[RawAlert]:
        """Flushes and returns all currently active deduplicated alerts."""
        with self._lock:
            alerts = list(self._active_window.values())
            self._active_window.clear()
            self._last_emission_count.clear()
            return alerts

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns operational deduplication statistics."""
        with self._lock:
            reduction_ratio = (
                self._total_suppressed / self._total_ingested
                if self._total_ingested > 0 else 0.0
            )
            return {
                "total_ingested": self._total_ingested,
                "total_suppressed": self._total_suppressed,
                "reduction_ratio": round(reduction_ratio, 4),
                "active_window_keys": len(self._active_window)
            }
