from __future__ import annotations
"""
AHRAS Statistical Engine — Publication Level
--------------------------------------------
Detects statistical and behavioral anomalies through per-entity baseline analysis
and multi-mechanism scoring across 5 OCSF telemetry classes:
  - network_activity
  - process_activity
  - file_activity
  - cloud_api
  - network_conn

Mechanisms Implemented:
  a) Z-Score Baseline with Effective Std Floor (T1030 / T1071)
  b) EWMA Baseline (slow vs fast adaptation) (T1071)
  c) Temporal Density / Burst Detection with Absolute Count Floor (T1499 / T1498)
  d) Circadian Rhythm Anomaly (-log2 P(hour|entity) with Laplace smoothing) (T1078)
  e) Resource / Port Affinity & Rarity Scoring (T1046 / T1068)
  f) Behavioral Vector Drift (ΔD_behavioral for Risk Formula) (T1562)
  g) Inter-Arrival Time Regularity / C2 Beaconing (Coefficient of Variation) (T1071 / T1571)
  h) Per-Entity Hourly Data Volume Buckets (Exfiltration Detection) (T1048 / T1020)
  i) Bounded Entity Lifecycle (LRU + TTL Eviction + Storage Persistence)
  j) Alert Cooldown & Suppression Window
  k) Analyst Feedback API (mark_false_positive, reset_entity)
  l) Full Inspection / Profile Extraction API (get_entity_profile, get_all_profiles, get_stats)
  m) Defensible MITRE ATT&CK Mapping
"""

import math
import time
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict

import numpy as np

log = logging.getLogger(__name__)

# ── Tunable Parameters ────────────────────────────────────────────────────────
_WINDOW_SEC          = 300      # 5-minute sliding window for density
_ZSCORE_THRESHOLD    = 3.0      # std deviations to flag
_EWMA_ALPHA          = 0.05     # slow smoothing factor
_EWMA_FAST_ALPHA     = 0.20     # fast smoothing factor
_DENSITY_THRESHOLD   = 100      # events per window
_DENSITY_ABS_FLOOR   = 50       # absolute count trigger floor for young entities
_DRIFT_THRESHOLD     = 2.5      # Euclidean distance for behavioral drift
_MIN_HISTORY         = 20       # minimum observations before full scoring
_MAX_ENTITIES        = 10_000   # cap for bounded memory
_ENTITY_TTL_SEC      = 3600     # idle eviction (1 hour)
_COOLDOWN_SEC        = 60       # alert suppression window per (entity, mechanism)


# ─────────────────────────────────────────────────────────────────────────────
# Safe Accessor Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(d: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely access deeply nested dictionary fields without triggering
    None.get() or None.lower() TypeErrors.
    """
    curr = d
    for k in keys:
        if not isinstance(curr, dict):
            return default
        curr = curr.get(k)
        if curr is None:
            return default
    return curr


def _extract_timestamp(evt: dict) -> float:
    """Extract event timestamp safely; falls back to time.time()."""
    t = evt.get("event_time") or evt.get("time")
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    return time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Result Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StatResult:
    """
    Structured anomaly scoring output from the statistical engine.
    Exposes primary scores for downstream hybrid fusion and evaluation harness.
    """
    entity_key:          str
    is_anomaly:          bool
    confidence:          float          # Overall confidence score [0.0, 1.0]
    zscore:              float          # Z-score metric deviation
    ewma_deviation:      float          # EWMA relative deviation
    temporal_density:    int            # Event count in sliding window
    behavioral_drift:    float          # ΔD_behavioral (Euclidean distance)
    flags:               list[str] = field(default_factory=list)
    baseline_n:          int = 0
    circadian_anomaly:   float = 0.0    # -log2 P(hour|entity)
    affinity_score:      float = 0.0    # Resource novelty / rarity score
    beacon_regularity:   float = 0.0    # 1.0 - normalized CV (high = periodic)
    volume_score:        float = 0.0    # Data exfiltration ratio
    cooldown_suppressed: bool = False   # True if duplicate alert suppressed
    mitre_techniques:    list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Per-Entity Behavioral State
# ─────────────────────────────────────────────────────────────────────────────

class _EntityState:
    """
    Maintains baseline statistics, histograms, vectors, and affinity sets
    for a single tracked entity (e.g. IP address, username, hostname).
    """
    __slots__ = (
        "_values", "_timestamps", "_ewma", "_ewma_fast", "_ewma_var",
        "_centroid", "_n", "_last_seen", "_hourly_counts",
        "_seen_resources", "_inter_arrival_gaps", "_volume_buckets",
        "_is_suppressed",
    )

    def __init__(self):
        self._values:             deque = deque(maxlen=500)
        self._timestamps:         deque = deque(maxlen=5000)
        self._ewma:               Optional[float] = None
        self._ewma_fast:          Optional[float] = None
        self._ewma_var:           float = 0.0
        self._centroid:           Optional[np.ndarray] = None
        self._n:                  int = 0
        self._last_seen:          float = time.time()
        self._hourly_counts:      list[int] = [0] * 24  # Circadian 24-hr histogram
        self._seen_resources:     set[str] = set()       # Resource affinity
        self._inter_arrival_gaps: deque = deque(maxlen=100) # Gaps between events
        self._volume_buckets:     defaultdict[int, float] = defaultdict(float) # Hour bucket -> bytes
        self._is_suppressed:      bool = False

    def update_value(self, v: float) -> None:
        self._values.append(v)
        self._last_seen = time.time()
        self._n += 1

        if self._ewma is None:
            self._ewma      = v
            self._ewma_fast = v
            self._ewma_var  = 0.0
        else:
            diff             = v - self._ewma
            self._ewma      += _EWMA_ALPHA * diff
            self._ewma_fast += _EWMA_FAST_ALPHA * (v - self._ewma_fast)
            self._ewma_var   = (1 - _EWMA_ALPHA) * (self._ewma_var + _EWMA_ALPHA * (diff ** 2))

    def update_vector(self, vec: np.ndarray) -> None:
        if self._centroid is None:
            self._centroid = vec.copy().astype(np.float64)
        else:
            self._centroid = (1 - _EWMA_ALPHA) * self._centroid + _EWMA_ALPHA * vec
        self._last_seen = time.time()

    def log_timestamp(self, ts: float) -> None:
        if self._timestamps:
            gap = ts - self._timestamps[-1]
            if gap >= 0.001:
                self._inter_arrival_gaps.append(gap)
        self._timestamps.append(ts)
        self._last_seen = time.time()

        # Update circadian hour histogram (derived from event timestamp)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        self._hourly_counts[dt.hour] += 1

    def log_volume(self, ts: float, volume_bytes: float) -> None:
        hour_bucket = int(ts // 3600)
        self._volume_buckets[hour_bucket] += volume_bytes

    # ── Mechanism (a): Z-Score with Effective Std Floor ──────────────────────

    def zscore(self, v: float) -> float:
        """
        Z-score metric deviation.
        Includes effective std floor max(std, mean * 0.05, 1e-3) to prevent
        std=0 degeneracy when baseline values are uniform.
        """
        if len(self._values) < _MIN_HISTORY:
            return 0.0
        arr  = np.array(self._values, dtype=np.float64)
        mean = float(arr.mean())
        std  = float(arr.std())
        eff_std = max(std, abs(mean) * 0.05, 1e-3)
        return abs(v - mean) / eff_std

    # ── Mechanism (b): EWMA Deviation ───────────────────────────────────────

    def ewma_deviation(self, v: float) -> float:
        if self._ewma is None or self._n < _MIN_HISTORY:
            return 0.0
        denom = max(abs(self._ewma), 1e-3)
        return abs(v - self._ewma) / denom

    # ── Mechanism (c): Temporal Density & Burst ──────────────────────────────

    def events_in_window(self, ts: float, window_sec: float = _WINDOW_SEC) -> int:
        cutoff = ts - window_sec
        return sum(1 for t in self._timestamps if t >= cutoff)

    # ── Mechanism (d): Circadian Rhythm Anomaly ──────────────────────────────

    def circadian_anomaly(self, ts: float) -> float:
        """
        Calculates information-theoretic anomaly score: -log2 P(hour|entity)
        with Laplace (+1) smoothing.
        Higher values (> 5.0) indicate activity during unusual hours.
        """
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        total = sum(self._hourly_counts)
        if total < _MIN_HISTORY:
            return 0.0
        prob = (self._hourly_counts[hour] + 1.0) / (total + 24.0)
        return float(-math.log2(prob))

    # ── Mechanism (e): Resource / Port Affinity ──────────────────────────────

    def check_affinity(self, resource_key: str) -> tuple[float, bool]:
        """
        Checks resource key against entity history.
        Returns (affinity_score, is_first_use).
        """
        if not resource_key or resource_key == "unknown":
            return (0.0, False)
        is_first_use = resource_key not in self._seen_resources
        self._seen_resources.add(resource_key)
        if self._n < _MIN_HISTORY:
            return (0.0, False)
        if is_first_use:
            score = 1.0 if len(self._seen_resources) > 5 else 0.5
            return (score, True)
        return (0.0, False)

    # ── Mechanism (f): Behavioral Drift (ΔD_behavioral) ──────────────────────

    def behavioral_drift(self, vec: np.ndarray) -> float:
        if self._centroid is None or self._n < _MIN_HISTORY:
            return 0.0
        return float(np.linalg.norm(vec - self._centroid))

    # ── Mechanism (g): Inter-Arrival Regularity (C2 Beaconing) ────────────────

    def beacon_regularity(self) -> float:
        """
        Calculates 1 - min(CV / 0.5, 1.0) on inter-event time gaps.
        Low Coefficient of Variation (CV < 0.2) indicates periodic C2 beaconing.
        """
        if len(self._inter_arrival_gaps) < 10:
            return 0.0
        gaps = np.array(self._inter_arrival_gaps, dtype=np.float64)
        mean_gap = float(gaps.mean())
        if mean_gap < 0.05:  # ignore sub-50ms bursts
            return 0.0
        std_gap = float(gaps.std())
        cv = std_gap / mean_gap
        if cv < 0.3:
            return float(1.0 - min(cv / 0.5, 1.0))
        return 0.0

    # ── Mechanism (h): Data Exfiltration Volume Score ───────────────────────

    def volume_score(self, ts: float) -> float:
        """
        Compares current 1-hour volume bucket against historical hourly mean volume.
        Returns ratio anomaly score if current hour exceeds 4x historical average.
        """
        if len(self._volume_buckets) < 3:
            return 0.0
        current_bucket = int(ts // 3600)
        past_volumes = [v for b, v in self._volume_buckets.items() if b < current_bucket]
        if not past_volumes:
            return 0.0
        avg_vol = float(np.mean(past_volumes))
        if avg_vol < 1024:  # ignore minimal traffic (< 1 KB)
            return 0.0
        curr_vol = self._volume_buckets[current_bucket]
        ratio = curr_vol / avg_vol
        return float(ratio) if ratio > 4.0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Engine Main Class
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalEngine:
    """
    Production Statistical Engine for AHRAS.
    Implements 13 statistical & behavioral detection mechanisms.
    Thread-safe via single RLock.
    """

    def __init__(self):
        self._entities: Dict[tuple, _EntityState] = {}
        self._lock = threading.RLock()
        self._n_scored = 0
        self._cooldowns: Dict[tuple, float] = {}   # (entity_key, mechanism) -> ts
        self._global_resource_counts: Dict[str, int] = defaultdict(int)

        # Background daemon thread for TTL eviction & maintenance
        self._evict_thread = threading.Thread(
            target=self._evict_loop, daemon=True, name="stat-evict"
        )
        self._evict_thread.start()

    def _get_entity(self, cls: str, key: str) -> _EntityState:
        ek = (cls, key)
        if ek not in self._entities:
            if len(self._entities) >= _MAX_ENTITIES:
                self._evict_oldest()
            self._entities[ek] = _EntityState()
        return self._entities[ek]

    def _evict_oldest(self) -> None:
        if not self._entities:
            return
        oldest = min(self._entities, key=lambda k: self._entities[k]._last_seen)
        del self._entities[oldest]

    def _evict_loop(self) -> None:
        while True:
            time.sleep(300)
            cutoff = time.time() - _ENTITY_TTL_SEC
            with self._lock:
                stale = [k for k, v in self._entities.items() if v._last_seen < cutoff]
                for k in stale:
                    del self._entities[k]
                # Cleanup expired cooldown entries
                now = time.time()
                expired_cd = [k for k, t in self._cooldowns.items() if now - t > _COOLDOWN_SEC]
                for k in expired_cd:
                    del self._cooldowns[k]

    # ── Main Scoring Method ───────────────────────────────────────────────────

    def score(self, evt: dict, vec: np.ndarray | None) -> StatResult:
        """
        Score an incoming OCSF event against per-entity baseline.
        Returns a StatResult with detection flags and MITRE ATT&CK mapping.
        """
        if not isinstance(evt, dict):
            evt = {}

        cls        = _get(evt, "ocsf_class", default="other")
        entity_key = _extract_entity_key(evt)
        metric_val = _extract_metric_value(evt)
        ts         = _extract_timestamp(evt)
        res_key    = _extract_resource_key(evt)
        vol_bytes  = _extract_volume_bytes(evt)

        if vec is None:
            from detection.feature_extractor import feature_dim
            vec = np.zeros(feature_dim(cls), dtype=np.float64)

        with self._lock:
            state = self._get_entity(cls, entity_key)

            # Check if entity is marked false positive by analyst
            if state._is_suppressed:
                return StatResult(
                    entity_key=entity_key, is_anomaly=False, confidence=0.0,
                    zscore=0.0, ewma_deviation=0.0, temporal_density=0,
                    behavioral_drift=0.0, flags=["analyst_suppressed"],
                    baseline_n=state._n, cooldown_suppressed=True
                )

            # Update timeline & baselines
            state.log_timestamp(ts)
            state.log_volume(ts, vol_bytes)
            state.update_value(metric_val)
            state.update_vector(vec)
            self._global_resource_counts[res_key] += 1

            # Compute mechanism scores
            z               = state.zscore(metric_val)
            ewma            = state.ewma_deviation(metric_val)
            dens            = state.events_in_window(ts)
            circadian       = state.circadian_anomaly(ts)
            aff_score, is_first = state.check_affinity(res_key)
            drift           = state.behavioral_drift(vec)
            beacon          = state.beacon_regularity()
            vol_score       = state.volume_score(ts)
            n               = state._n

        # ── Evaluate Mechanism Flags & MITRE Mappings ─────────────────────────
        flags: list[str] = []
        mitre: list[str] = []

        # (a) Z-score
        if z > _ZSCORE_THRESHOLD:
            flags.append(f"z_score={z:.1f}")
            mitre.append("T1030")

        # (b) EWMA
        if ewma > 1.5:
            flags.append(f"ewma_dev={ewma:.2f}")
            mitre.append("T1071")

        # (c) Temporal density & burst (paired with absolute floor)
        if dens >= _DENSITY_ABS_FLOOR or dens > _DENSITY_THRESHOLD:
            flags.append(f"density={dens}/window")
            mitre.append("T1499")

        # (d) Circadian rhythm
        if circadian > 5.5:
            flags.append(f"circadian_offhours={circadian:.1f}")
            mitre.append("T1078")

        # (e) Resource affinity & rarity
        if is_first and aff_score > 0.5:
            flags.append(f"new_resource={res_key}")
            mitre.append("T1046")

        # (f) Behavioral drift (ΔD_behavioral)
        if drift > _DRIFT_THRESHOLD:
            flags.append(f"drift={drift:.2f}")
            mitre.append("T1562")

        # (g) C2 Beaconing
        if beacon > 0.7:
            flags.append(f"c2_beacon={beacon:.2f}")
            mitre.append("T1071")

        # (h) Data Volume Exfiltration
        if vol_score > 4.0:
            flags.append(f"exfil_vol={vol_score:.1f}x")
            mitre.append("T1048")

        # Deduplicate MITRE list
        mitre = sorted(list(set(mitre)))

        # ── (j) Cooldown Suppression Check ────────────────────────────────────
        cooldown_suppressed = False
        if flags:
            now_time = time.time()
            primary_mech = flags[0].split("=")[0]
            cd_key = (entity_key, primary_mech)
            with self._lock:
                last_fired = self._cooldowns.get(cd_key, 0.0)
                if now_time - last_fired < _COOLDOWN_SEC:
                    cooldown_suppressed = True
                else:
                    self._cooldowns[cd_key] = now_time

        # ── Confidence Scoring ────────────────────────────────────────────────
        z_norm         = min(z / (_ZSCORE_THRESHOLD * 2), 1.0)
        ewma_norm      = min(ewma / 3.0, 1.0)
        density_norm   = min(dens / (_DENSITY_THRESHOLD * 2), 1.0)
        drift_norm     = min(drift / (_DRIFT_THRESHOLD * 2), 1.0)
        circadian_norm = min(circadian / 8.0, 1.0)
        beacon_norm    = beacon

        confidence = (0.25 * z_norm +
                      0.20 * ewma_norm +
                      0.15 * density_norm +
                      0.15 * drift_norm +
                      0.15 * circadian_norm +
                      0.10 * beacon_norm)
        confidence = round(float(np.clip(confidence, 0, 1.0)), 4)

        is_anomaly = (len(flags) >= 1) and not cooldown_suppressed

        self._n_scored += 1

        return StatResult(
            entity_key=entity_key,
            is_anomaly=is_anomaly,
            confidence=confidence,
            zscore=round(z, 3),
            ewma_deviation=round(ewma, 3),
            temporal_density=dens,
            behavioral_drift=round(drift, 4),
            flags=flags,
            baseline_n=n,
            circadian_anomaly=round(circadian, 3),
            affinity_score=round(aff_score, 3),
            beacon_regularity=round(beacon, 3),
            volume_score=round(vol_score, 3),
            cooldown_suppressed=cooldown_suppressed,
            mitre_techniques=mitre,
        )

    # ── (k) Analyst Feedback API ──────────────────────────────────────────────

    def mark_false_positive(self, cls: str, entity_key: str) -> None:
        """Suppress alerts for an entity marked as false positive by SOC analyst."""
        with self._lock:
            st = self._get_entity(cls, entity_key)
            st._is_suppressed = True
            log.info(f"[STAT] Marked false positive for ({cls}, {entity_key})")

    def reset_entity(self, cls: str, entity_key: str) -> None:
        """Reset baseline state for an entity."""
        with self._lock:
            ek = (cls, entity_key)
            if ek in self._entities:
                del self._entities[ek]
            log.info(f"[STAT] Reset entity baseline for ({cls}, {entity_key})")

    # ── (l) Inspection & Reporting API ────────────────────────────────────────

    def get_entity_profile(self, cls: str, entity_key: str) -> dict:
        with self._lock:
            ek = (cls, entity_key)
            if ek not in self._entities:
                return {"entity_key": entity_key, "exists": False}
            st = self._entities[ek]
            return {
                "entity_key":       entity_key,
                "ocsf_class":       cls,
                "exists":           True,
                "baseline_n":       st._n,
                "last_seen":        st._last_seen,
                "ewma":             st._ewma,
                "seen_resources":   list(st._seen_resources),
                "hourly_histogram": list(st._hourly_counts),
                "is_suppressed":    st._is_suppressed,
            }

    def get_all_profiles(self) -> list[dict]:
        with self._lock:
            return [self.get_entity_profile(c, k) for (c, k) in list(self._entities.keys())]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "tracked_entities": len(self._entities),
                "total_scored":     self._n_scored,
                "active_cooldowns": len(self._cooldowns),
            }

    # ── (i) Snapshot Persistence Methods ──────────────────────────────────────

    def save_snapshot(self) -> int:
        """Persist entity baselines to storage via storage/store.py's store."""
        try:
            from storage.store import get_store
            store = get_store()
            saved = 0
            with self._lock:
                for (cls, key), st in self._entities.items():
                    doc = {
                        "type": "stat_snapshot",
                        "ocsf_class": cls,
                        "entity_key": key,
                        "baseline_n": st._n,
                        "ewma": st._ewma,
                        "hourly_counts": list(st._hourly_counts),
                        "resources": list(st._seen_resources),
                        "saved_at": time.time(),
                    }
                    store.insert("stat_snapshots", doc)
                    saved += 1
            return saved
        except Exception as e:
            log.warning(f"[STAT] Snapshot save failed: {e}")
            return 0

    def load_snapshot(self) -> int:
        """Reload entity baselines from storage."""
        try:
            from storage.store import get_store
            store = get_store()
            docs = store.query("stat_snapshots", {"type": "stat_snapshot"})
            loaded = 0
            with self._lock:
                for d in docs:
                    cls = d.get("ocsf_class")
                    key = d.get("entity_key")
                    if cls and key:
                        st = self._get_entity(cls, key)
                        st._n = d.get("baseline_n", 0)
                        st._ewma = d.get("ewma")
                        if "hourly_counts" in d:
                            st._hourly_counts = list(d["hourly_counts"])
                        if "resources" in d:
                            st._seen_resources = set(d["resources"])
                        loaded += 1
            return loaded
        except Exception as e:
            log.warning(f"[STAT] Snapshot load failed: {e}")
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions: Key, Metric, Resource, Volume Extractors
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entity_key(evt: dict) -> str:
    cls = _get(evt, "ocsf_class", default="")
    if cls == "network_activity":
        return _get(evt, "src_endpoint", "ip", default="unknown")
    if cls in ("process_activity", "file_activity"):
        return _get(evt, "device", "hostname", default="unknown")
    if cls == "cloud_api":
        return _get(evt, "actor", "user", "name", default="unknown")
    if cls == "network_conn":
        return _get(evt, "dst_endpoint", "ip", default="unknown")
    return "unknown"


def _extract_metric_value(evt: dict) -> float:
    cls = _get(evt, "ocsf_class", default="")
    if cls == "network_activity":
        dur = float(_get(evt, "traffic", "duration_sec", default=1) or 1)
        pkts = float(_get(evt, "traffic", "packets", default=0) or 0)
        return pkts / max(dur, 0.001)

    if cls == "process_activity":
        return float(_get(evt, "actor", "process", "pid", default=0) or 0)

    if cls == "file_activity":
        return float(_get(evt, "enrichment", "entropy", default=0) or 0)

    if cls == "cloud_api":
        from detection.feature_extractor import _CLOUD_ACTIONS
        action = _get(evt, "api", "operation", default="")
        return float(_CLOUD_ACTIONS.get(action, 5))

    if cls == "network_conn":
        return float(_get(evt, "dst_endpoint", "port", default=0) or 0)

    return 0.0


def _extract_resource_key(evt: dict) -> str:
    cls = _get(evt, "ocsf_class", default="")
    if cls in ("network_activity", "network_conn"):
        port = _get(evt, "dst_endpoint", "port") or _get(evt, "traffic", "dst_port")
        return f"port:{port}" if port is not None else "port:unknown"
    if cls == "process_activity":
        pname = _get(evt, "process", "name") or _get(evt, "file", "name")
        return f"proc:{pname}" if pname else "proc:unknown"
    if cls == "file_activity":
        path = _get(evt, "file", "path") or _get(evt, "file", "name") or ""
        prefix = path.split("/")[1] if "/" in path and len(path.split("/")) > 1 else path
        return f"file:{prefix}" if prefix else "file:unknown"
    if cls == "cloud_api":
        op = _get(evt, "api", "operation")
        return f"api:{op}" if op else "api:unknown"
    return "resource:unknown"


def _extract_volume_bytes(evt: dict) -> float:
    cls = _get(evt, "ocsf_class", default="")
    if cls in ("network_activity", "network_conn"):
        b_out = _get(evt, "traffic", "bytes_out") or _get(evt, "traffic", "bytes") or 0
        return float(b_out)
    if cls == "file_activity":
        sz = _get(evt, "file", "size") or _get(evt, "enrichment", "bytes_written") or 0
        return float(sz)
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Instance & Module Entry Points
# ─────────────────────────────────────────────────────────────────────────────

_engine_instance: Optional[StatisticalEngine] = None
_singleton_lock  = threading.Lock()


def get_statistical_engine() -> StatisticalEngine:
    global _engine_instance
    with _singleton_lock:
        if _engine_instance is None:
            _engine_instance = StatisticalEngine()
    return _engine_instance


def run_statistical_engine(evt: dict, vec: np.ndarray | None) -> StatResult:
    """Convenience entry point for scoring events via the singleton StatisticalEngine."""
    return get_statistical_engine().score(evt, vec)
