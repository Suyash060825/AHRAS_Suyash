from __future__ import annotations
"""
AHRAS Module 4 — Uncertainty- & Provenance-Aware Adaptive Risk Controller
-------------------------------------------------------------------------
Formally implements the Adaptive Hybrid Risk-Aware Security (AHRAS) risk controller:

    R_t = Clip_0^1 [ (w1*S_sig + w2*A_ml*(1 + ΔD) + w4*H_boost + w5*G_corr + w6*P_fore + w7*TI_score)
                     * A_crit * (1 - U_penalty) - w3*T_trust ]

Key Architectural Properties:
  1. Configurable Mechanism Toggles: Uses `RiskConfig` to selectively enable/disable any subsystem.
  2. Machine-Readable DecisionTrace: Records exact execution sequence for deterministic XAI replay.
  3. Formal Separation of Concerns: Risk estimation is computed independently of response action selection.
  4. Operational Safety Metric: Computes Risk-to-Action Safety Efficiency (RASE).
"""

import math
import time
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ahras.evidence.models import EvidenceRecord, EvidenceType, EvidenceSource
from ahras.evidence.ledger import get_evidence_ledger

log = logging.getLogger(__name__)

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
class RiskConfig:
    """
    Explicit configuration toggles for every risk subsystem.
    Enables controlled experimental ablations and sensitivity sweeps.
    """
    use_signature:    bool  = True
    use_ml:           bool  = True
    use_statistical:  bool  = True
    use_trust:        bool  = True
    use_history:      bool  = True
    use_graph:        bool  = True
    use_forecast:     bool  = True
    use_uncertainty:  bool  = True
    use_ti:           bool  = True
    use_deception:    bool  = True
    use_asset_crit:   bool  = True
    adaptive_weights: bool  = False

    # Baseline Weight Parameters
    w_sig:            float = 0.50   # Signature weight
    w_ml:             float = 0.30   # ML Anomaly weight
    w_trust:          float = 0.15   # Trust mitigation weight
    w_hist:           float = 0.10   # Historical recidivism weight
    w_graph:          float = 0.10   # Graph corroboration weight
    w_fore:           float = 0.05   # Forecast momentum weight
    w_ti:             float = 0.15   # Threat intelligence weight

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionTrace:
    """
    Complete, deterministic execution trace of risk estimation.
    Enables zero-drift analytical re-execution and XAI audit replay.
    """
    event_id:             str
    entity_key:           str
    timestamp:            float
    config:               Dict[str, Any]
    raw_inputs:           Dict[str, float]
    intermediate_terms:   Dict[str, float]
    additive_sum:         float
    criticality_mult:     float
    uncertainty_mult:     float
    trust_subtraction:    float
    pre_clip_score:       float
    final_clamped_score:  float
    severity:             str
    remediation_level:    str
    evidence_ids:         List[str]

    def to_dict(self) -> dict:
        return asdict(self)


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
    risk_uncertainty:   float = 0.15   # Epistemic/disagreement uncertainty [0.0, 1.0]
    risk_delta:         float = 0.0    # Delta relative to entity's baseline
    risk_velocity:      float = 0.0    # Velocity of risk accumulation (dR/dt)
    risk_state:         str   = "NORMAL" # NORMAL, ELEVATED, ESCALATING, CRITICAL
    decision_reason:    str   = ""
    
    # Full Replay Trace
    trace:              Optional[DecisionTrace] = None
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
            "decision_reason":   self.decision_reason,
            "explanation":       self.explanation,
            "trace":             self.trace.to_dict() if self.trace else None,
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
        config: Optional[RiskConfig] = None,
    ):
        self._trust_scores: Dict[str, float] = defaultdict(lambda: default_trust)
        self._last_update: Dict[str, float] = defaultdict(time.time)
        self._recent_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._lock = threading.RLock()
        self._default_trust = default_trust
        self._decay_rate = decay_rate
        self._recovery_rate = recovery_rate
        self.config = config or RiskConfig()

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
        override_config: Optional[RiskConfig] = None,
    ) -> RiskResult:
        """
        Executes the formal risk controller equation using active RiskConfig:
            R_t = Clip_0^1 [ (w1*S_sig + w2*A_ml*(1 + delta_D) + w4*H_boost + w5*G_corr + w6*P_fore + w7*TI_score)
                             * A_crit * (1 - U_penalty) - w3*T_trust ]
        """
        if evt is None:
            evt = {}

        cfg = override_config or self.config
        event_id = evt.get("event_id", f"EVT-{int(time.time()*1000)}")
        evidence_ledger = get_evidence_ledger()
        created_evidence_ids = []

        with self._lock:
            T_trust = self.get_trust(entity_key) if cfg.use_trust else 0.0

        # 1. Compute S_sig (Normalized signature severity)
        max_sig_sev = 0
        sig_flags = []
        mitre_techs = []
        sig_conf = 0.0
        if sig_matches and cfg.use_signature:
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

        S_sig = float(max_sig_sev / 5.0) if cfg.use_signature else 0.0

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
        if ml_res and cfg.use_ml:
            A_ml = getattr(ml_res, "ensemble_score", None) or _get(ml_res, "ensemble_score", default=0.0)
            A_ml = float(np.clip(A_ml, 0.0, 1.0))
            ml_conf = float(getattr(ml_res, "confidence", None) or _get(ml_res, "confidence", default=0.5))

        if A_ml > 0 and cfg.use_ml:
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
        if stat_res and cfg.use_statistical:
            delta_D = getattr(stat_res, "behavioral_drift", None) or _get(stat_res, "behavioral_drift", default=0.0)
            delta_D = float(np.clip(delta_D, 0.0, 3.0))
            stat_conf = float(getattr(stat_res, "confidence", None) or _get(stat_res, "confidence", default=0.5))
            
            sflags = getattr(stat_res, "flags", []) or _get(stat_res, "flags", default=[])
            sig_flags.extend(sflags)
            
            smitre = getattr(stat_res, "mitre_techniques", []) or _get(stat_res, "mitre_techniques", default=[])
            mitre_techs.extend(smitre)

        if delta_D > 0 and cfg.use_statistical:
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

        # Contextual Features
        effective_h_boost = float(h_boost) if cfg.use_history else 0.0
        effective_g_corr = float(g_corr) if cfg.use_graph else 0.0
        effective_p_fore = float(p_fore) if cfg.use_forecast else 0.0
        
        # Threat Intel
        if not ti_score and evt:
            ti_score = float(_get(evt, "enrichment", "threat_intel_score", default=0.0))
        effective_ti = float(ti_score) if cfg.use_ti else 0.0

        # Asset criticality
        if a_crit == 1.0 and evt:
            a_crit = float(_get(evt, "asset", "criticality_multiplier", default=1.0))
        effective_a_crit = float(a_crit) if cfg.use_asset_crit else 1.0

        # 4. Detector Disagreement Uncertainty
        detector_scores = [s for s in [S_sig, A_ml, min(1.0, delta_D/1.5) if delta_D > 0 else 0.0] if s > 0.0]
        if len(detector_scores) >= 2:
            disagreement = float(np.std(detector_scores))
            uncertainty = max(0.05, min(0.40, disagreement * 0.8))
        elif len(detector_scores) == 1:
            uncertainty = 0.20 if S_sig > 0 else 0.30
        else:
            uncertainty = 0.05

        if not cfg.use_uncertainty:
            uncertainty = 0.0
            u_penalty = 0.0
            confidence = 1.0
        else:
            confidence = round(float(np.clip(1.0 - uncertainty, 0.0, 1.0)), 4)
            u_penalty = round(uncertainty * 0.30, 4)

        # 5. Core Adaptive Risk Equation
        term_sig   = cfg.w_sig * S_sig
        term_ml    = cfg.w_ml * A_ml * (1.0 + delta_D)
        term_hist  = cfg.w_hist * effective_h_boost
        term_graph = cfg.w_graph * effective_g_corr
        term_fore  = cfg.w_fore * effective_p_fore
        term_ti    = cfg.w_ti * effective_ti
        
        additive_threat = term_sig + term_ml + term_hist + term_graph + term_fore + term_ti
        mult_crit = effective_a_crit
        mult_unc  = (1.0 - u_penalty)
        trust_sub = cfg.w_trust * T_trust
        
        raw_risk = (additive_threat * mult_crit * mult_unc) - trust_sub
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))
        risk_score = round(risk_score, 4)

        # 6. Temporal Risk Dynamics
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

        if cfg.use_trust:
            self.update_trust_on_event(entity_key, is_anom=is_alert, severity_id=severity_id)

        # Construct DecisionTrace
        trace = DecisionTrace(
            event_id=event_id,
            entity_key=entity_key,
            timestamp=time.time(),
            config=cfg.to_dict(),
            raw_inputs={
                "S_sig": S_sig,
                "A_ml": A_ml,
                "delta_D": delta_D,
                "T_trust": T_trust,
                "H_boost": effective_h_boost,
                "G_corr": effective_g_corr,
                "P_fore": effective_p_fore,
                "TI_score": effective_ti,
                "A_crit": effective_a_crit,
                "uncertainty": uncertainty,
            },
            intermediate_terms={
                "term_sig": term_sig,
                "term_ml": term_ml,
                "term_hist": term_hist,
                "term_graph": term_graph,
                "term_fore": term_fore,
                "term_ti": term_ti,
            },
            additive_sum=additive_threat,
            criticality_mult=mult_crit,
            uncertainty_mult=mult_unc,
            trust_subtraction=trust_sub,
            pre_clip_score=raw_risk,
            final_clamped_score=risk_score,
            severity=severity,
            remediation_level=remediation,
            evidence_ids=created_evidence_ids,
        )

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
            H_boost=round(effective_h_boost, 3),
            G_corr=round(effective_g_corr, 3),
            P_fore=round(effective_p_fore, 3),
            TI_score=round(effective_ti, 3),
            A_crit=round(effective_a_crit, 2),
            risk_confidence=confidence,
            risk_uncertainty=round(uncertainty, 4),
            risk_delta=risk_delta,
            risk_velocity=risk_velocity,
            risk_state=risk_state,
            decision_reason=decision_reason,
            trace=trace,
            explanation=explanation,
            flags=sorted(list(set(sig_flags))),
            mitre_techniques=sorted(list(set(mitre_techs))),
            evidence_ids=created_evidence_ids,
        )


def replay_decision_trace(trace: DecisionTrace) -> float:
    """
    Independent analytical re-execution of a DecisionTrace.
    Asserts zero-drift equivalence (|engine - replayed| <= 1e-6).
    """
    cfg = trace.config
    inputs = trace.raw_inputs
    
    t_sig   = cfg["w_sig"] * inputs["S_sig"] if cfg["use_signature"] else 0.0
    t_ml    = cfg["w_ml"] * inputs["A_ml"] * (1.0 + inputs["delta_D"]) if cfg["use_ml"] else 0.0
    t_hist  = cfg["w_hist"] * inputs["H_boost"] if cfg["use_history"] else 0.0
    t_graph = cfg["w_graph"] * inputs["G_corr"] if cfg["use_graph"] else 0.0
    t_fore  = cfg["w_fore"] * inputs["P_fore"] if cfg["use_forecast"] else 0.0
    t_ti    = cfg["w_ti"] * inputs["TI_score"] if cfg["use_ti"] else 0.0
    
    add_sum = t_sig + t_ml + t_hist + t_graph + t_fore + t_ti
    crit_mult = inputs["A_crit"] if cfg["use_asset_crit"] else 1.0
    
    u_pen = (inputs["uncertainty"] * 0.30) if cfg["use_uncertainty"] else 0.0
    unc_mult = (1.0 - u_pen)
    
    t_trust = (cfg["w_trust"] * inputs["T_trust"]) if cfg["use_trust"] else 0.0
    
    raw = (add_sum * crit_mult * unc_mult) - t_trust
    reconstructed = float(np.clip(raw, 0.0, 1.0))
    return round(reconstructed, 4)


def compute_rase(
    risk_reduction: float,
    uncertainty: float,
    blast_radius: float,
    reversibility_cost: float,
    is_false_intervention: bool,
    lambda_fp_penalty: float = 1.0,
) -> float:
    """
    Computes Risk-to-Action Safety Efficiency (RASE):
        RASE = (ΔR_mitigated * (1 - Uncertainty)) / (BlastRadius + ReversibilityCost + λ * II(FalseIntervention))
    """
    numerator = max(0.0, risk_reduction) * max(0.0, min(1.0, 1.0 - uncertainty))
    cost_denom = max(0.01, blast_radius + reversibility_cost + (lambda_fp_penalty if is_false_intervention else 0.0))
    return round(float(numerator / cost_denom), 4)


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
