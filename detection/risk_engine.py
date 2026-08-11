from __future__ import annotations
"""
AHRAS Module 4 — Adaptive Risk Engine
------------------------------------
Formally implements the Adaptive Hybrid Risk-Aware Security (AHRAS) risk scoring formula:

    R_t = w1 * S_sig + w2 * A_ml * (1 + ΔD_behavioral) - w3 * T_trust

where:
    - S_sig ∈ [0.0, 1.0]           : Normalized signature severity (sev / 5.0)
    - A_ml ∈ [0.0, 1.0]            : ML ensemble anomaly score
    - ΔD_behavioral ∈ [0.0, 3.0]   : Behavioral vector drift Euclidean distance
    - T_trust ∈ [0.0, 1.0]         : Dynamic entity trust score (decays on alert, recovers on clean activity)
    - Weights                      : w1 = 0.50, w2 = 0.30, w3 = 0.15

Severity Classification & Action Escalation:
    - INFO     : R_t < 0.30     -> LOG_ONLY
    - LOW      : 0.30 <= R_t < 0.50 -> LOG_ONLY
    - MEDIUM   : 0.50 <= R_t < 0.70 -> STAGE_APPROVAL
    - HIGH     : 0.70 <= R_t < 0.90 -> SOC_ALERT_HIGH
    - CRITICAL : R_t >= 0.90    -> AUTO_REMEDIATE
"""

import math
import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Default Formula Weights
W1_SIG   = 0.50   # Signature weight
W2_ML    = 0.30   # ML Anomaly weight
W3_TRUST = 0.15   # Trust mitigation weight

# Severity Thresholds
THRESHOLD_LOW      = 0.30
THRESHOLD_MEDIUM   = 0.50
THRESHOLD_HIGH     = 0.70
THRESHOLD_CRITICAL = 0.90


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested dictionary accessor."""
    curr = d
    for k in keys:
        if not isinstance(curr, dict):
            return default
        curr = curr.get(k)
        if curr is None:
            return default
    return curr


@dataclass
class RiskResult:
    """Output from the Adaptive Risk Engine."""
    entity_key:         str
    risk_score:         float          # R_t ∈ [0.0, 1.0]
    severity:           str            # INFO, LOW, MEDIUM, HIGH, CRITICAL
    severity_id:        int            # 1=INFO, 2=LOW, 3=MEDIUM, 4=HIGH, 5=CRITICAL
    remediation_level:  str            # LOG_ONLY, STAGE_APPROVAL, SOC_ALERT_HIGH, AUTO_REMEDIATE
    is_alert:           bool
    
    # Formula Components
    S_sig:              float          # Normalized signature severity
    A_ml:               float          # ML ensemble anomaly score
    delta_D:            float          # Behavioral vector drift
    T_trust:            float          # Dynamic entity trust score
    
    explanation:        str
    flags:              list[str] = field(default_factory=list)
    mitre_techniques:   list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_key":        self.entity_key,
            "risk_score":        self.risk_score,
            "severity":          self.severity,
            "severity_id":       self.severity_id,
            "remediation_level": self.remediation_level,
            "is_alert":          self.is_alert,
            "components": {
                "S_sig":   self.S_sig,
                "A_ml":    self.A_ml,
                "delta_D": self.delta_D,
                "T_trust": self.T_trust,
            },
            "explanation":       self.explanation,
            "flags":             self.flags,
            "mitre_techniques":  self.mitre_techniques,
        }


class AdaptiveRiskEngine:
    """
    Tracks dynamic entity trust scores and computes adaptive risk scores.
    Thread-safe via RLock.
    """

    def __init__(
        self,
        default_trust: float = 0.50,
        decay_rate: float = 0.15,
        recovery_rate: float = 0.02,
    ):
        self._trust_scores: Dict[str, float] = defaultdict(lambda: default_trust)
        self._last_update: Dict[str, float] = defaultdict(time.time)
        self._lock = threading.RLock()
        self._default_trust = default_trust
        self._decay_rate = decay_rate
        self._recovery_rate = recovery_rate

    def get_trust(self, entity_key: str) -> float:
        with self._lock:
            return round(float(self._trust_scores[entity_key]), 4)

    def set_trust(self, entity_key: str, trust_level: float) -> None:
        with self._lock:
            self._trust_scores[entity_key] = float(np.clip(trust_level, 0.0, 1.0))
            self._last_update[entity_key] = time.time()
            log.info(f"[RISK] Set trust for '{entity_key}' to {trust_level:.2f}")

    def update_trust_on_event(self, entity_key: str, is_anom: bool, severity_id: int = 1) -> float:
        """
        Decays trust if an anomaly occurs; recovers trust slowly on normal events.
        """
        with self._lock:
            curr_t = self._trust_scores[entity_key]
            if is_anom:
                penalty = self._decay_rate * (severity_id / 3.0)
                curr_t = max(0.0, curr_t - penalty)
            else:
                curr_t = min(1.0, curr_t + self._recovery_rate)
            self._trust_scores[entity_key] = curr_t
            self._last_update[entity_key] = time.time()
            return round(float(curr_t), 4)

    def score_risk(
        self,
        entity_key: str,
        sig_matches: list,
        ml_res: Any,
        stat_res: Any,
        evt: dict = None
    ) -> RiskResult:
        """
        Evaluates R_t = w1*S_sig + w2*A_ml*(1 + delta_D) - w3*T_trust
        """
        if evt is None:
            evt = {}

        with self._lock:
            T_trust = self.get_trust(entity_key)

        # 1. Compute S_sig (Highest signature severity normalized to [0, 1])
        max_sig_sev = 0
        sig_flags = []
        mitre_techs = []
        if sig_matches:
            for match in sig_matches:
                sev = getattr(match, "severity", None) or _get(match, "severity", default=1)
                if isinstance(sev, str):
                    smap = {"INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}
                    sev = smap.get(sev.upper(), 1)
                max_sig_sev = max(max_sig_sev, int(sev))
                
                rname = getattr(match, "rule_name", None) or _get(match, "rule_name", default="rule")
                sig_flags.append(f"sig:{rname}")
                
                mtech = getattr(match, "mitre_technique", None) or getattr(match, "mitre_id", None) or _get(match, "mitre_technique") or _get(match, "mitre_id")
                if mtech:
                    mitre_techs.append(mtech)

        S_sig = float(max_sig_sev / 5.0)

        # 2. Compute A_ml (ML Ensemble Score)
        A_ml = 0.0
        if ml_res:
            A_ml = getattr(ml_res, "ensemble_score", None) or _get(ml_res, "ensemble_score", default=0.0)
            A_ml = float(np.clip(A_ml, 0.0, 1.0))

        # 3. Compute delta_D (Behavioral Vector Drift)
        delta_D = 0.0
        if stat_res:
            delta_D = getattr(stat_res, "behavioral_drift", None) or _get(stat_res, "behavioral_drift", default=0.0)
            delta_D = float(np.clip(delta_D, 0.0, 3.0))
            
            sflags = getattr(stat_res, "flags", []) or _get(stat_res, "flags", default=[])
            sig_flags.extend(sflags)
            
            smitre = getattr(stat_res, "mitre_techniques", []) or _get(stat_res, "mitre_techniques", default=[])
            mitre_techs.extend(smitre)

        # 4. Evaluate Adaptive Risk Formula: R_t = w1*S_sig + w2*A_ml*(1 + delta_D) - w3*T_trust
        raw_risk = (W1_SIG * S_sig) + (W2_ML * A_ml * (1.0 + delta_D)) - (W3_TRUST * T_trust)
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))
        risk_score = round(risk_score, 4)

        # 5. Severity & Remediation Action Mapping
        if risk_score >= THRESHOLD_CRITICAL:
            severity = "CRITICAL"
            severity_id = 5
            remediation = "AUTO_REMEDIATE"
            is_alert = True
        elif risk_score >= THRESHOLD_HIGH:
            severity = "HIGH"
            severity_id = 4
            remediation = "SOC_ALERT_HIGH"
            is_alert = True
        elif risk_score >= THRESHOLD_MEDIUM:
            severity = "MEDIUM"
            severity_id = 3
            remediation = "STAGE_APPROVAL"
            is_alert = True
        elif risk_score >= THRESHOLD_LOW:
            severity = "LOW"
            severity_id = 2
            remediation = "LOG_ONLY"
            is_alert = False
        else:
            severity = "INFO"
            severity_id = 1
            remediation = "LOG_ONLY"
            is_alert = False

        # Update dynamic entity trust based on risk evaluation
        self.update_trust_on_event(entity_key, is_anom=is_alert, severity_id=severity_id)

        # Deduplicate flags and MITRE techniques
        sig_flags = sorted(list(set(sig_flags)))
        mitre_techs = sorted(list(set(mitre_techs)))

        # Explanation narrative
        explanation = (
            f"Adaptive Risk R_t={risk_score:.3f} [{severity}] for '{entity_key}'. "
            f"Components: S_sig={S_sig:.2f}, A_ml={A_ml:.2f}, ΔD={delta_D:.2f}, T_trust={T_trust:.2f}. "
            f"Remediation policy: {remediation}."
        )

        return RiskResult(
            entity_key=entity_key,
            risk_score=risk_score,
            severity=severity,
            severity_id=severity_id,
            remediation_level=remediation,
            is_alert=is_alert,
            S_sig=round(S_sig, 3),
            A_ml=round(A_ml, 3),
            delta_D=round(delta_D, 3),
            T_trust=round(T_trust, 3),
            explanation=explanation,
            flags=sig_flags,
            mitre_techniques=mitre_techs,
        )


# Singleton
_risk_engine_instance: Optional[AdaptiveRiskEngine] = None
_risk_lock = threading.Lock()


def get_risk_engine() -> AdaptiveRiskEngine:
    global _risk_engine_instance
    with _risk_lock:
        if _risk_engine_instance is None:
            _risk_engine_instance = AdaptiveRiskEngine()
    return _risk_engine_instance


def run_risk_engine(
    entity_key: str,
    sig_matches: list,
    ml_res: Any,
    stat_res: Any,
    evt: dict = None
) -> RiskResult:
    """Convenience entry point for scoring risk via singleton AdaptiveRiskEngine."""
    return get_risk_engine().score_risk(entity_key, sig_matches, ml_res, stat_res, evt)
