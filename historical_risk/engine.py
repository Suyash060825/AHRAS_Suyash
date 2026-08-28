from __future__ import annotations
"""
AHRAS Historical Risk Engine — Recidivism-Based Threat Memory
--------------------------------------------------------------
Maintains persistent memory of past incidents, alerts, and risk scores across
all monitored indicators (IPs, users, hostnames, file hashes).

Recidivism Formula:
  incident_boost = min(30, incident_count * 2)
  alert_boost    = min(15, alert_count)
  recency_factor = 1.0 (< 7 days) | 0.5 (7-30 days) | 0.25 (> 30 days)
  history_boost  = (incident_boost + alert_boost) * recency_factor
"""

import time
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


@dataclass
class IndicatorHistory:
    indicator:        str
    incident_count:   int = 0
    alert_count:      int = 0
    first_seen:       float = field(default_factory=time.time)
    last_seen:        float = field(default_factory=time.time)
    recent_events:    List[dict] = field(default_factory=list)
    risk_scores:      List[float] = field(default_factory=list)
    tags:             List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "indicator":      self.indicator,
            "incident_count": self.incident_count,
            "alert_count":    self.alert_count,
            "first_seen":     self.first_seen,
            "last_seen":      self.last_seen,
            "total_scored":   len(self.risk_scores),
            "recent_scores":  self.risk_scores[-10:],
            "tags":           self.tags,
        }


class HistoricalRiskEngine:
    """
    Tracks indicator recidivism across events and computes contextual history boosts.
    Thread-safe.
    """

    def __init__(self, max_history_per_indicator: int = 100):
        self._history: Dict[str, IndicatorHistory] = {}
        self._max_history = max_history_per_indicator
        self._lock = threading.RLock()

    def record_event(self, indicator: str, risk_score: float, is_alert: bool = False,
                     is_incident: bool = False, event_dict: Optional[dict] = None) -> None:
        if not indicator:
            return

        with self._lock:
            now = time.time()
            if indicator not in self._history:
                self._history[indicator] = IndicatorHistory(
                    indicator=indicator,
                    first_seen=now,
                    last_seen=now,
                )
            hist = self._history[indicator]
            hist.last_seen = now
            hist.risk_scores.append(float(risk_score))
            if len(hist.risk_scores) > self._max_history:
                hist.risk_scores = hist.risk_scores[-self._max_history:]

            if is_alert:
                hist.alert_count += 1
            if is_incident:
                hist.incident_count += 1

            if event_dict:
                evt_snap = {
                    "time": now,
                    "risk_score": float(risk_score),
                    "is_alert": is_alert,
                }
                hist.recent_events.append(evt_snap)
                if len(hist.recent_events) > self._max_history:
                    hist.recent_events = hist.recent_events[-self._max_history:]

    def compute_history_boost(self, indicator: str, normalized_unit_scale: bool = True) -> float:
        """
        Computes history boost from recidivism formula.
        If normalized_unit_scale is True, returns boost in [0.0, 0.45], else [0.0, 45.0].
        """
        if not indicator:
            return 0.0

        with self._lock:
            hist = self._history.get(indicator)
            if not hist:
                return 0.0

            incident_boost = min(30.0, float(hist.incident_count) * 2.0)
            alert_boost = min(15.0, float(hist.alert_count))
            
            elapsed_days = (time.time() - hist.last_seen) / 86400.0
            if elapsed_days < 7.0:
                recency = 1.0
            elif elapsed_days < 30.0:
                recency = 0.50
            else:
                recency = 0.25

            boost_100 = min(45.0, (incident_boost + alert_boost) * recency)
            if normalized_unit_scale:
                return round(boost_100 / 100.0, 4)
            return round(boost_100, 2)

    def get_indicator_history(self, indicator: str) -> Optional[IndicatorHistory]:
        with self._lock:
            return self._history.get(indicator)

    def get_all_histories(self, limit: int = 100) -> List[dict]:
        with self._lock:
            return [h.to_dict() for h in list(self._history.values())[:limit]]

    def count_tracked(self) -> int:
        with self._lock:
            return len(self._history)


# Singleton instance
_history_engine_instance: Optional[HistoricalRiskEngine] = None
_history_lock = threading.Lock()


def get_historical_risk_engine() -> HistoricalRiskEngine:
    global _history_engine_instance
    with _history_lock:
        if _history_engine_instance is None:
            _history_engine_instance = HistoricalRiskEngine()
    return _history_engine_instance
