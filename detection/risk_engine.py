from __future__ import annotations
"""
AHRAS Module 4 — Uncertainty- & Provenance-Aware Adaptive Risk Controller
-------------------------------------------------------------------------
Formally implements the Adaptive Hybrid Risk-Aware Security (AHRAS) risk controller:

    R_t = Clip_0^1 [ (w1*S_sig + w2*A_ml*(1 + ΔD) + w4*H_boost + w5*G_corr + w6*P_fore + w7*TI_score)
                     * A_crit * (1 - U_penalty) - w3*T_trust ]

where:
    - S_sig ∈ [0.0, 1.0]          : Normalized signature severity (sev / 5.0)
    - A_ml ∈ [0.0, 1.0]           : Calibrated ML ensemble anomaly probability
    - ΔD_behavioral ∈ [0.0, 3.0]  : Behavioral vector drift Euclidean distance
    - T_trust ∈ [0.0, 1.0]        : Dynamic entity trust score (decays on alert, recovers on clean activity)
    - H_boost ∈ [0.0, 0.3]        : Historical recidivism boost from repeated past incidents
    - G_corr ∈ [0.0, 0.3]         : Graph episode corroboration & lateral movement factor
    - P_fore ∈ [0.0, 0.2]         : Predictive risk escalation momentum from early warning forecaster
    - TI_score ∈ [0.0, 0.4]       : Verified threat intelligence IOC match score
    - A_crit ∈ [0.8, 1.5]         : Asset / Identity criticality multiplier
    - U_penalty ∈ [0.0, 0.3]      : Epistemic & aleatoric uncertainty dampening factor
    - Baseline Weights            : w1 = 0.50, w2 = 0.30, w3 = 0.15

Uncertainty & Calibration:
    Risk confidence and uncertainty are explicitly quantified from detector agreement,
    sample coverage, and variance across heterogeneous evidence sources.
"""

import math
import time
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ahras.evidence.models import EvidenceRecord, EvidenceType, EvidenceSource
from ahras.evidence.ledger import get_evidence_ledger

log = logging.getLogger(__name__)

# Default Formula Weights
W1_SIG    = 0.50   # Signature weight
W2_ML     = 0.30   # ML Anomaly weight
W3_TRUST  = 0.15   # Trust mitigation weight
W4_HIST   = 0.10   # Historical recidivism weight
W5_GRAPH  = 0.10   # Graph correlation weight
W6_FORE   = 0.05   # Forecast momentum weight
W7_TI     = 0.15   # Threat intelligence weight

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
    """Output from the Uncertainty- & Provenance-Aware Adaptive Risk Controller."""
    entity_key:         str
    risk_score:         float          # R_t ∈ [0.0, 1.0]
    severity:           str            # INFO, LOW, MEDIUM, HIGH, CRITICAL
    severity_id:        int            # 1=INFO, 2=LOW, 3=MEDIUM, 4=HIGH, 5=CRITICAL
    remediation_level:  str            # LOG_ONLY, STAGE_APPROVAL, SOC_ALERT_HIGH, AUTO_REMEDIATE
    is_alert:           bool
    
    # Formula Components (for exact XAI analytical reconstruction)
    S_sig:              float          # Normalized signature severity
    A_ml:               float          # ML ensemble anomaly score
    delta_D:            float          # Behavioral vector drift
    T_trust:            float          # Dynamic entity trust score
    H_boost:            float = 0.0    # Historical recidivism component
    G_corr:             float = 0.0    # Graph correlation component
    P_fore:             float = 0.0    # Forecast early warning component
    TI_score:           float = 0.0    # Threat intelligence component
    A_crit:             float = 1.0    # Asset criticality multiplier
    
    # Uncertainty & Dynamics
    risk_confidence:    float = 0.85   # Confidence in risk evaluation [0.0, 1.0]
    risk_uncertainty:   float = 0.15   # Epistemic/aleatoric uncertainty [0.0, 1.0]
    risk_delta:         float = 0.0    # Delta relative to entity's baseline
    risk_velocity:      float = 0.0    # Velocity of risk accumulation (dR/dt)
    risk_state:         str   = "NORMAL" # NORMAL, ELEVATED, ESCALATING, CRITICAL
    decision_reason:    str   = ""
    
    explanation:        str   = ""
    flags:              list[str] = field(default_factory=list)
    mitre_techniques:   list[str] = field(default_factory=list)
    evidence_ids:       list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_key":        self.entity_key,
            "risk_score":        self.risk_score,
            "severity":          self.severity,
            "severity_id":       self.severity_id,
            "remediation_level": self.remediation_level,
            "is_alert":          self.is_alert,
            "components": {
                "S_sig":    self.S_sig,
                "A_ml":     self.A_ml,
                "delta_D":  self.delta_D,
                "T_trust":  self.T_trust,
                "H_boost":  self.H_boost,
                "G_corr":   self.G_corr,
                "P_fore":   self.P_fore,
                "TI_score": self.TI_score,
                "A_crit":   self.A_crit,
            },
            "uncertainty": {
                "confidence":  self.risk_confidence,
                "uncertainty": self.risk_uncertainty,
                "delta":       self.risk_delta,
                "velocity":    self.risk_velocity,
                "state":       self.risk_state,
            },
            "explanation":       self.explanation,
            "decision_reason":   self.decision_reason,
            "flags":             self.flags,
            "mitre_techniques":  self.mitre_techniques,
            "evidence_ids":      self.evidence_ids,
        }


class AdaptiveRiskEngine:
    """
    Tracks dynamic entity trust, historical risk state, and uncertainty.
    Converts heterogeneous detection evidence into calibrated risk decisions.
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
        self._recent_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
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
        evt: dict = None,
        h_boost: float = 0.0,
        g_corr: float = 0.0,
        p_fore: float = 0.0,
        ti_score: float = 0.0,
        a_crit: float = 1.0,
    ) -> RiskResult:
        """
        Evaluates the full uncertainty-aware risk controller equation:
            R_t = Clip_0^1 [ (w1*S_sig + w2*A_ml*(1 + delta_D) + w4*H_boost + w5*G_corr + w6*P_fore + w7*TI_score)
                             * A_crit * (1 - U_penalty) - w3*T_trust ]
        """
        if evt is None:
            evt = {}

        event_id = evt.get("event_id", f"EVT-{int(time.time()*1000)}")
        evidence_ledger = get_evidence_ledger()
        created_evidence_ids = []

        with self._lock:
            T_trust = self.get_trust(entity_key)

        # 1. Compute S_sig (Normalized signature severity)
        max_sig_sev = 0
        sig_flags = []
        mitre_techs = []
        sig_conf = 0.0
        if sig_matches:
            for match in sig_matches:
                sev = getattr(match, "severity", None) or _get(match, "severity", default=1)
                if isinstance(sev, str):
                    smap = {"INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}
                    sev = smap.get(sev.upper(), 1)
                max_sig_sev = max(max_sig_sev, int(sev))
                
                m_conf = float(getattr(match, "confidence", None) or _get(match, "confidence", default=0.9))
                sig_conf = max(sig_conf, m_conf)
                
                rname = getattr(match, "rule_name", None) or _get(match, "rule_name", default="rule")
                sig_flags.append(f"sig:{rname}")
                
                mtech = getattr(match, "mitre_technique", None) or getattr(match, "mitre_id", None) or _get(match, "mitre_technique") or _get(match, "mitre_id")
                if mtech:
                    mitre_techs.append(mtech)

        S_sig = float(max_sig_sev / 5.0)

        if S_sig > 0:
            ev_sig = EvidenceRecord(
                event_id=event_id,
                entity_id=entity_key,
                source=EvidenceSource.SURICATA_ENGINE.value,
                detector_type=EvidenceType.SIGNATURE.value,
                raw_score=float(max_sig_sev),
                normalized_score=S_sig,
                confidence=sig_conf or 0.95,
                uncertainty=round(1.0 - (sig_conf or 0.95), 4),
                mitre_mapping=list(mitre_techs),
                explanation=f"Signature rule triggered with severity {max_sig_sev}/5",
            )
            evidence_ledger.record_evidence(ev_sig)
            created_evidence_ids.append(ev_sig.evidence_id)

        # 2. Compute A_ml (Calibrated ML Ensemble Score)
        A_ml = 0.0
        ml_conf = 0.5
        if ml_res:
            A_ml = getattr(ml_res, "ensemble_score", None) or _get(ml_res, "ensemble_score", default=0.0)
            A_ml = float(np.clip(A_ml, 0.0, 1.0))
            ml_conf = float(getattr(ml_res, "confidence", None) or _get(ml_res, "confidence", default=0.5))

        if A_ml > 0:
            ev_ml = EvidenceRecord(
                event_id=event_id,
                entity_id=entity_key,
                source=EvidenceSource.AUTOENCODER.value,
                detector_type=EvidenceType.ML_ANOMALY.value,
                raw_score=A_ml,
                normalized_score=A_ml,
                confidence=ml_conf,
                uncertainty=round(1.0 - ml_conf, 4),
                explanation=f"ML ensemble anomaly score {A_ml:.3f} (confidence: {ml_conf:.2f})",
            )
            evidence_ledger.record_evidence(ev_ml)
            created_evidence_ids.append(ev_ml.evidence_id)

        # 3. Compute delta_D (Behavioral Vector Drift)
        delta_D = 0.0
        stat_conf = 0.5
        if stat_res:
            delta_D = getattr(stat_res, "behavioral_drift", None) or _get(stat_res, "behavioral_drift", default=0.0)
            delta_D = float(np.clip(delta_D, 0.0, 3.0))
            stat_conf = float(getattr(stat_res, "confidence", None) or _get(stat_res, "confidence", default=0.5))
            
            sflags = getattr(stat_res, "flags", []) or _get(stat_res, "flags", default=[])
            sig_flags.extend(sflags)
            
            smitre = getattr(stat_res, "mitre_techniques", []) or _get(stat_res, "mitre_techniques", default=[])
            mitre_techs.extend(smitre)

        if delta_D > 0:
            ev_stat = EvidenceRecord(
                event_id=event_id,
                entity_id=entity_key,
                source=EvidenceSource.WELFORD_STAT_ENGINE.value,
                detector_type=EvidenceType.STATISTICAL_DRIFT.value,
                raw_score=delta_D,
                normalized_score=min(1.0, delta_D / 3.0),
                confidence=stat_conf,
                uncertainty=round(1.0 - stat_conf, 4),
                explanation=f"Statistical behavioral drift ΔD={delta_D:.2f}",
            )
            evidence_ledger.record_evidence(ev_stat)
            created_evidence_ids.append(ev_stat.evidence_id)

        # Extract Threat Intel from event if available
        if not ti_score and evt:
            ti_score = float(_get(evt, "enrichment", "threat_intel_score", default=0.0))

        # Asset criticality weighting
        if a_crit == 1.0 and evt:
            a_crit = float(_get(evt, "asset", "criticality_multiplier", default=1.0))

        # 4. Uncertainty Estimation (Epistemic disagreement + Aleatoric noise)
        detector_scores = [s for s in [S_sig, A_ml, min(1.0, delta_D/1.5) if delta_D > 0 else 0.0] if s > 0.0]
        if len(detector_scores) >= 2:
            disagreement = float(np.std(detector_scores))
            uncertainty = max(0.05, min(0.40, disagreement * 0.8))
        elif len(detector_scores) == 1:
            uncertainty = 0.20 if S_sig > 0 else 0.30
        else:
            uncertainty = 0.05

        confidence = round(float(np.clip(1.0 - uncertainty, 0.0, 1.0)), 4)
        u_penalty = round(uncertainty * 0.30, 4) # Max 12% dampening for high uncertainty

        # 5. Core Adaptive Risk Equation
        base_threat = (W1_SIG * S_sig) + (W2_ML * A_ml * (1.0 + delta_D))
        contextual_threat = (W4_HIST * h_boost) + (W5_GRAPH * g_corr) + (W6_FORE * p_fore) + (W7_TI * ti_score)
        
        raw_risk = (base_threat + contextual_threat) * a_crit * (1.0 - u_penalty) - (W3_TRUST * T_trust)
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))
        risk_score = round(risk_score, 4)

        # 6. Temporal Risk Dynamics (Velocity & Delta)
        with self._lock:
            history = self._recent_scores[entity_key]
            if history:
                mean_hist = float(np.mean(history))
                risk_delta = round(risk_score - mean_hist, 4)
                risk_velocity = round((risk_score - history[-1]) / max(0.1, time.time() - self._last_update[entity_key]), 4)
            else:
                risk_delta = 0.0
                risk_velocity = 0.0
            history.append(risk_score)

        # 7. Severity & Remediation Action Mapping
        if risk_score >= THRESHOLD_CRITICAL:
            severity = "CRITICAL"
            severity_id = 5
            remediation = "AUTO_REMEDIATE"
            risk_state = "CRITICAL"
            is_alert = True
        elif risk_score >= THRESHOLD_HIGH:
            severity = "HIGH"
            severity_id = 4
            remediation = "SOC_ALERT_HIGH"
            risk_state = "ESCALATING"
            is_alert = True
        elif risk_score >= THRESHOLD_MEDIUM:
            severity = "MEDIUM"
            severity_id = 3
            remediation = "STAGE_APPROVAL"
            risk_state = "ELEVATED"
            is_alert = True
        elif risk_score >= THRESHOLD_LOW:
            severity = "LOW"
            severity_id = 2
            remediation = "LOG_ONLY"
            risk_state = "MONITORING"
            is_alert = False
        else:
            severity = "INFO"
            severity_id = 1
            remediation = "LOG_ONLY"
            risk_state = "NORMAL"
            is_alert = False

        # Update dynamic entity trust
        self.update_trust_on_event(entity_key, is_anom=is_alert, severity_id=severity_id)

        # Deduplicate flags and MITRE techniques
        sig_flags = sorted(list(set(sig_flags)))
        mitre_techs = sorted(list(set(mitre_techs)))

        # Formal Decision Reason & Explanation
        decision_reason = (
            f"Evaluated risk={risk_score:.3f} (Conf={confidence:.2f}, Uncert={uncertainty:.2f}) "
            f"under state {risk_state}. S_sig={S_sig:.2f}, A_ml={A_ml:.2f}, ΔD={delta_D:.2f}, "
            f"T_trust={T_trust:.2f}. Remediation: {remediation}."
        )

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
            H_boost=round(h_boost, 3),
            G_corr=round(g_corr, 3),
            P_fore=round(p_fore, 3),
            TI_score=round(ti_score, 3),
            A_crit=round(a_crit, 2),
            risk_confidence=confidence,
            risk_uncertainty=round(uncertainty, 4),
            risk_delta=risk_delta,
            risk_velocity=risk_velocity,
            risk_state=risk_state,
            decision_reason=decision_reason,
            explanation=explanation,
            flags=sig_flags,
            mitre_techniques=mitre_techs,
            evidence_ids=created_evidence_ids,
        )


# Singleton instance
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
    evt: dict = None,
    **kwargs,
) -> RiskResult:
    """Convenience entry point for scoring risk via singleton AdaptiveRiskEngine."""
    return get_risk_engine().score_risk(entity_key, sig_matches, ml_res, stat_res, evt, **kwargs)
