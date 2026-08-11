from __future__ import annotations
"""
AHRAS Hybrid Detection Engine — Combiner
-----------------------------------------
Orchestrates all three detection engines and combines their outputs
into a single, unified DetectionResult.

Decision logic:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Engine fired?  │  Signature  │  ML Anomaly  │  Statistical     │
  ├─────────────────┼─────────────┼──────────────┼──────────────────┤
  │  Confidence     │   0.90      │    0.72      │   0.61           │
  │  Weight         │   0.50      │    0.30      │   0.20           │
  └──────────────────────────────────────────────────────────────────┘

  Final confidence = weighted sum of fired engines.
  is_alert = True if:
    • Any signature fires  (known attack — always alert)
    • OR (ML fires AND confidence > 0.65)
    • OR (all 3 fire with any confidence)
    • OR (ensemble confidence > 0.72)

Severity mapping:
  confidence < 0.50 → INFO
  0.50 – 0.65       → LOW
  0.65 – 0.80       → MEDIUM
  0.80 – 0.90       → HIGH
  > 0.90            → CRITICAL
"""

import uuid
import time
import logging
import threading
import numpy as np

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from detection.feature_extractor import extract, feature_names as get_feature_names
from detection.signature_engine.rules import run_signature_engine, SignatureMatch
from detection.anomaly_engine.ml_engine import run_anomaly_engine, AnomalyResult
from detection.statistical_engine.stat_engine import run_statistical_engine, StatResult
from detection.xai_explainer import explain

log = logging.getLogger(__name__)

# ── Engine weights (must sum to 1.0) ─────────────────────────────────────────
_W_SIG  = 0.50
_W_ML   = 0.30
_W_STAT = 0.20

# ── Confidence thresholds ─────────────────────────────────────────────────────
_ALERT_THRESHOLD = 0.50   # minimum confidence to create an alert
_HIGH_CONF       = 0.80
_CRITICAL_CONF   = 0.90

_SEVERITY_MAP = [
    (0.90, "CRITICAL"),
    (0.80, "HIGH"),
    (0.65, "MEDIUM"),
    (0.50, "LOW"),
    (0.00, "INFO"),
]


def _severity_label(conf: float) -> str:
    for threshold, label in _SEVERITY_MAP:
        if conf >= threshold:
            return label
    return "INFO"


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    # Identity
    detection_id:   str
    event_id:       str
    ocsf_class:     str
    time:           str

    # Verdict
    is_alert:       bool
    confidence:     float          # 0–1
    severity:       str            # INFO / LOW / MEDIUM / HIGH / CRITICAL
    attack_type:    str            # best label from fired engines
    engines_fired:  list[str]      # subset of ["signature","anomaly","statistical"]

    # Per-engine detail
    signature_matches: list[dict]  # serialized SignatureMatch objects
    anomaly_result:    dict        # serialized AnomalyResult
    stat_result:       dict        # serialized StatResult

    # XAI
    explanation:    dict           # from xai_explainer

    # Original normalized event (pass-through for dashboard)
    normalized_event: dict = field(default_factory=dict)

    # Processing metadata
    processing_ms:  float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sig_to_dict(m: SignatureMatch) -> dict:
    return {
        "rule_id":        m.rule_id,
        "rule_name":      m.rule_name,
        "attack_type":    m.attack_type,
        "severity":       m.severity,
        "confidence":     m.confidence,
        "description":    m.description,
        "mitre_tactic":   m.mitre_tactic,
        "mitre_technique":m.mitre_technique,
        "evidence":       m.evidence,
    }


def _anom_to_dict(r: AnomalyResult) -> dict:
    return {
        "is_anomaly":            r.is_anomaly,
        "confidence":            r.confidence,
        "isolation_score":       r.isolation_score,
        "reconstruction_error":  r.reconstruction_error,
        "svm_score":             r.svm_score,
        "ensemble_score":        r.ensemble_score,
        "n_models_fired":        r.n_models_fired,
        "model_trained":         r.model_trained,
    }


def _stat_to_dict(r: StatResult) -> dict:
    return {
        "entity_key":          r.entity_key,
        "is_anomaly":          r.is_anomaly,
        "confidence":          r.confidence,
        "zscore":              r.zscore,
        "ewma_deviation":      r.ewma_deviation,
        "temporal_density":    r.temporal_density,
        "behavioral_drift":    r.behavioral_drift,
        "flags":               r.flags,
        "baseline_n":          r.baseline_n,
        "circadian_anomaly":   getattr(r, "circadian_anomaly", 0.0),
        "affinity_score":      getattr(r, "affinity_score", 0.0),
        "beacon_regularity":   getattr(r, "beacon_regularity", 0.0),
        "volume_score":        getattr(r, "volume_score", 0.0),
        "cooldown_suppressed": getattr(r, "cooldown_suppressed", False),
        "mitre_techniques":    getattr(r, "mitre_techniques", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combiner
# ─────────────────────────────────────────────────────────────────────────────

class HybridCombiner:
    """
    Thread-safe combiner that runs all three engines and fuses results.
    Maintains per-class feature baselines for XAI deviation calculation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Rolling baseline per OCSF class for XAI deviation calculation
        # key: ocsf_class → running mean of feature vectors
        self._baselines: dict[str, np.ndarray] = {}
        self._baseline_n: dict[str, int]        = {}
        self._n_processed = 0
        self._n_alerts    = 0

    def _update_baseline(self, cls: str, vec: np.ndarray) -> np.ndarray:
        """Online update of feature vector baseline (running mean)."""
        with self._lock:
            if cls not in self._baselines:
                self._baselines[cls]   = vec.copy()
                self._baseline_n[cls]  = 1
            else:
                n = self._baseline_n[cls]
                self._baselines[cls]  = (self._baselines[cls] * n + vec) / (n + 1)
                self._baseline_n[cls] = n + 1
            return self._baselines[cls].copy()

    def process(self, evt: dict) -> Optional[DetectionResult]:
        """
        Run full hybrid detection pipeline on one OCSF-normalized event.
        Returns DetectionResult (always), or None if event class unsupported.
        """
        t0        = time.perf_counter()
        cls       = evt.get("ocsf_class", "")
        event_id  = evt.get("event_id", str(uuid.uuid4()))

        # ── Feature extraction ────────────────────────────────────────────────
        vec = extract(evt)
        if vec is None:
            return None

        feat_names   = get_feature_names(cls)
        baseline_vec = self._update_baseline(cls, vec)

        # ── Engine 1: Signatures ──────────────────────────────────────────────
        sig_matches   = run_signature_engine(evt)
        sig_fired     = len(sig_matches) > 0
        sig_conf      = max((m.confidence for m in sig_matches), default=0.0)
        sig_severity  = max((m.severity   for m in sig_matches), default=0)
        best_sig_type = sig_matches[0].attack_type if sig_matches else ""
        best_sig_mitre= sig_matches[0].mitre_technique if sig_matches else ""

        # ── Engine 2: ML Anomaly ──────────────────────────────────────────────
        ml_result  = run_anomaly_engine(cls, vec)
        ml_fired   = ml_result.is_anomaly
        ml_conf    = ml_result.confidence

        # ── Engine 3: Statistical ─────────────────────────────────────────────
        stat_result = run_statistical_engine(evt, vec)
        stat_fired  = stat_result.is_anomaly
        stat_conf   = stat_result.confidence

        # ── Ensemble combination ──────────────────────────────────────────────
        engines_fired = []
        weighted_conf = 0.0

        if sig_fired:
            engines_fired.append("signature")
            weighted_conf += _W_SIG * sig_conf

        if ml_fired:
            engines_fired.append("anomaly")
            weighted_conf += _W_ML * ml_conf

        if stat_fired:
            engines_fired.append("statistical")
            weighted_conf += _W_STAT * stat_conf

        # Boost confidence when multiple engines agree (corroborating evidence)
        n_engines = len(engines_fired)
        if n_engines == 2:
            weighted_conf = min(weighted_conf * 1.20, 1.0)
        elif n_engines == 3:
            weighted_conf = min(weighted_conf * 1.40, 1.0)

        confidence = round(weighted_conf, 4)

        # ── Alert decision ────────────────────────────────────────────────────
        is_alert = (
            sig_fired                                      # any signature = always alert
            or (ml_fired and confidence > 0.60)            # ML above threshold
            or (n_engines == 3)                            # all 3 agree
            or (confidence > _ALERT_THRESHOLD and n_engines >= 2)
        )

        # ── Attack type label ─────────────────────────────────────────────────
        if best_sig_type:
            attack_type = best_sig_type
        elif ml_fired:
            attack_type = "ml_anomaly"
        elif stat_fired:
            attack_type = "behavioral_anomaly"
        else:
            attack_type = ""

        # ── Severity ──────────────────────────────────────────────────────────
        # Severity derives from THREE sources, taking the maximum:
        #   1. Ensemble confidence mapped through _SEVERITY_MAP
        #   2. Highest signature rule severity (directly mapped: 5→CRITICAL etc.)
        #   3. OCSF event severity_id from normalizer
        _SIG_SEV_LABELS = {5:"CRITICAL", 4:"HIGH", 3:"MEDIUM", 2:"LOW", 1:"INFO", 0:"INFO"}
        _LABEL_RANK     = {"CRITICAL":5, "HIGH":4, "MEDIUM":3, "LOW":2, "INFO":1}
        conf_severity   = _severity_label(confidence) if is_alert else "INFO"
        sig_severity_label  = _SIG_SEV_LABELS.get(sig_severity, "INFO")
        ocsf_sev_label  = _SIG_SEV_LABELS.get(evt.get("severity_id", 1), "INFO")
        # Take the highest severity across all three sources
        severity = max(
            [conf_severity, sig_severity_label, ocsf_sev_label if is_alert else "INFO"],
            key=lambda x: _LABEL_RANK.get(x, 0)
        )

        # ── XAI Explanation ───────────────────────────────────────────────────
        # Get Isolation Forest pipeline if available (for IF-importance)
        if_pipe = None
        try:
            from detection.anomaly_engine.ml_engine import _get_detector
            det     = _get_detector(cls)
            if_pipe = det._if_pipe
        except Exception:
            pass

        xai = explain(
            vec=vec,
            feature_names=feat_names,
            ocsf_class=cls,
            attack_type=attack_type,
            sig_confidence=sig_conf,
            ml_confidence=ml_conf,
            stat_confidence=stat_conf,
            if_pipe=if_pipe,
            baseline_mean=baseline_vec,
        )

        # ── Add MITRE to explanation ──────────────────────────────────────────
        if best_sig_mitre:
            xai["mitre_technique"] = best_sig_mitre

        processing_ms = round((time.perf_counter() - t0) * 1000, 2)

        with self._lock:
            self._n_processed += 1
            if is_alert:
                self._n_alerts += 1

        return DetectionResult(
            detection_id    = str(uuid.uuid4()),
            event_id        = event_id,
            ocsf_class      = cls,
            time            = datetime.now(timezone.utc).isoformat(),
            is_alert        = is_alert,
            confidence      = confidence,
            severity        = severity,
            attack_type     = attack_type,
            engines_fired   = engines_fired,
            signature_matches = [_sig_to_dict(m) for m in sig_matches],
            anomaly_result  = _anom_to_dict(ml_result),
            stat_result     = _stat_to_dict(stat_result),
            explanation     = xai,
            normalized_event = evt,
            processing_ms   = processing_ms,
        )

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_processed": self._n_processed,
                "total_alerts":    self._n_alerts,
                "alert_rate":      round(self._n_alerts / max(self._n_processed, 1), 4),
                "classes_seen":    list(self._baselines.keys()),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_combiner_instance: Optional[HybridCombiner] = None
_combiner_lock = threading.Lock()


def get_combiner() -> HybridCombiner:
    global _combiner_instance
    with _combiner_lock:
        if _combiner_instance is None:
            _combiner_instance = HybridCombiner()
    return _combiner_instance
