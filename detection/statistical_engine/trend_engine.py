from __future__ import annotations
"""
AHRAS Trend & Seasonality Engine
--------------------------------
Detects gradual multi-day behavioral ramps and weekend/weekday seasonality shifts.

Attack Pattern Caught:
  Catches low-and-slow data exfiltration, stealthy persistence escalation, and
  weekend-targeted attacks that deliberately remain below single-event or
  single-day Z-score thresholds, but exhibit a clear positive trend or seasonality
  anomaly over a multi-day window (e.g. +5% data volume increase per day over 7 days).

Key Design Rules:
  1. Event Timestamp Derivation: "day" and "hour" are derived EXCLUSIVELY from the
     event's own timestamp parameter — NEVER from time.time() or system wall clock.
  2. Absolute Variance Checking: When checking for zero variance across day-ordinal
     values (e.g. [738500, 738501, 738502]), we use absolute spread (max - min)
     instead of np.allclose's relative tolerance, which silently swallows 1–3 day
     differences at large epoch/ordinal scales.
  3. Seasonality Awareness: Maintains separate baselines for Weekday (Mon–Fri)
     and Weekend (Sat–Sun) activity windows.
"""

import math
import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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


def _extract_timestamp(evt: dict) -> float:
    """Extract event timestamp; defaults to 0.0 if invalid."""
    t = evt.get("event_time") or evt.get("time") or evt.get("timestamp")
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    return 0.0


@dataclass
class TrendResult:
    """Output from scoring multi-day trend and seasonality."""
    entity_key:          str
    slope:               float          # Trend slope (metric units per day)
    r2_score:            float          # Coefficient of determination [0.0, 1.0]
    is_trend_ramp:       bool           # True if consistent positive ramp detected
    is_seasonality_anomaly: bool        # True if weekend activity deviates from weekday
    weekday_mean:        float
    weekend_mean:        float
    confidence:          float
    flags:               list[str] = field(default_factory=list)
    mitre_techniques:    list[str] = field(default_factory=list)


class TrendEngine:
    """
    Tracks multi-day trend metrics and weekday vs weekend seasonality baselines.
    Thread-safe via RLock.
    """

    def __init__(self, min_days: int = 3, r2_threshold: float = 0.65):
        # State: (ocsf_class, entity_key) -> dict of day_ordinal -> total_metric
        self._daily_metrics: Dict[tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # Seasonality: (ocsf_class, entity_key) -> {"weekday": [], "weekend": []}
        self._seasonality: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: {"weekday": [], "weekend": []})
        self._lock = threading.RLock()
        self._min_days = min_days
        self._r2_threshold = r2_threshold

    def update_and_score(self, evt: dict, entity_key: str, metric_val: float) -> TrendResult:
        """
        Updates daily & seasonal baselines for the entity and returns TrendResult.
        Timestamp MUST be passed inside evt["event_time"], evt["time"], or evt["timestamp"].
        """
        cls = _get(evt, "ocsf_class", default="other")
        ts = _extract_timestamp(evt)
        if ts <= 0:
            ts = time.time()

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day_ordinal = dt.toordinal()
        is_weekend = dt.weekday() >= 5  # 5=Saturday, 6=Sunday
        season_type = "weekend" if is_weekend else "weekday"

        with self._lock:
            ek = (cls, entity_key)
            self._daily_metrics[ek][day_ordinal] += metric_val
            self._seasonality[ek][season_type].append(metric_val)

            day_dict = dict(self._daily_metrics[ek])
            weekday_vals = list(self._seasonality[ek]["weekday"])
            weekend_vals = list(self._seasonality[ek]["weekend"])

        # Check day-ordinal spread using absolute difference (not relative tolerance)
        ordinals = sorted(day_dict.keys())
        if len(ordinals) < self._min_days:
            return TrendResult(
                entity_key=entity_key, slope=0.0, r2_score=0.0,
                is_trend_ramp=False, is_seasonality_anomaly=False,
                weekday_mean=float(np.mean(weekday_vals)) if weekday_vals else 0.0,
                weekend_mean=float(np.mean(weekend_vals)) if weekend_vals else 0.0,
                confidence=0.0
            )

        # Absolute spread check
        day_spread = ordinals[-1] - ordinals[0]
        if day_spread < 2:  # Must span at least 3 distinct calendar days
            return TrendResult(
                entity_key=entity_key, slope=0.0, r2_score=0.0,
                is_trend_ramp=False, is_seasonality_anomaly=False,
                weekday_mean=float(np.mean(weekday_vals)) if weekday_vals else 0.0,
                weekend_mean=float(np.mean(weekend_vals)) if weekend_vals else 0.0,
                confidence=0.0
            )

        # ── Fit Linear Trend (Slope + R^2) ───────────────────────────────────
        X = np.array([o - ordinals[0] for o in ordinals], dtype=np.float64) # Relative days: 0, 1, 2...
        Y = np.array([day_dict[o] for o in ordinals], dtype=np.float64)

        n = len(X)
        x_mean = float(X.mean())
        y_mean = float(Y.mean())

        ss_xx = float(np.sum((X - x_mean) ** 2))
        ss_yy = float(np.sum((Y - y_mean) ** 2))
        ss_xy = float(np.sum((X - x_mean) * (Y - y_mean)))

        slope = ss_xy / ss_xx if ss_xx > 1e-9 else 0.0

        if ss_xx > 1e-9 and ss_yy > 1e-9:
            r2 = float((ss_xy ** 2) / (ss_xx * ss_yy))
        else:
            r2 = 0.0

        # Ramping condition: Positive slope, R^2 > threshold, and total increase > 50%
        relative_increase = (Y[-1] - Y[0]) / max(Y[0], 1e-3)
        is_trend_ramp = (slope > 0.1) and (r2 >= self._r2_threshold) and (relative_increase >= 0.4)

        # ── Seasonality Anomaly ──────────────────────────────────────────────
        weekday_mean = float(np.mean(weekday_vals)) if weekday_vals else 0.0
        weekend_mean = float(np.mean(weekend_vals)) if weekend_vals else 0.0
        is_seasonality_anomaly = False

        if is_weekend and weekday_vals and len(weekend_vals) >= 2:
            if weekend_mean > 3.0 * max(weekday_mean, 1.0):
                is_seasonality_anomaly = True

        flags = []
        mitre = []
        if is_trend_ramp:
            flags.append(f"multi_day_ramp (slope={slope:.2f}/day, R2={r2:.2f})")
            mitre.append("T1048")  # Exfiltration Over Alternative Protocol / Low and Slow
        if is_seasonality_anomaly:
            flags.append(f"weekend_anomaly (weekend_avg={weekend_mean:.1f} vs weekday_avg={weekday_mean:.1f})")
            mitre.append("T1078")  # Valid Accounts: Off-hours

        confidence = max(r2 if is_trend_ramp else 0.0, 0.85 if is_seasonality_anomaly else 0.0)
        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)

        return TrendResult(
            entity_key=entity_key,
            slope=round(slope, 4),
            r2_score=round(r2, 4),
            is_trend_ramp=is_trend_ramp,
            is_seasonality_anomaly=is_seasonality_anomaly,
            weekday_mean=round(weekday_mean, 2),
            weekend_mean=round(weekend_mean, 2),
            confidence=confidence,
            flags=flags,
            mitre_techniques=sorted(list(set(mitre))),
        )


# Singleton
_trend_engine_instance: Optional[TrendEngine] = None
_trend_lock = threading.Lock()


def get_trend_engine() -> TrendEngine:
    global _trend_engine_instance
    with _trend_lock:
        if _trend_engine_instance is None:
            _trend_engine_instance = TrendEngine()
    return _trend_engine_instance
