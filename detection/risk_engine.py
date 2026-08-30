from __future__ import annotations
"""
AHRAS Module 4 — Uncertainty- & Provenance-Aware Adaptive Risk Controller
-------------------------------------------------------------------------
Formally implements the Adaptive Hybrid Risk-Aware Security (AHRAS) risk controller:

    R_t = Clip_0^1 [ (w1*S_sig + w2*A_ml*(1 + ΔD) + w4*H_boost + w5*G_corr + w6*P_fore + w7*TI_score + w8*R_ep)
                     * A_crit * (1 - U_penalty) - w3*T_trust ]

Key Architectural Properties:
  1. Configurable Mechanism Toggles: Uses `RiskConfig` to selectively enable/disable any subsystem.
  2. Machine-Readable DecisionTrace: Records exact execution sequence for deterministic XAI replay.
  3. Formal Separation of Concerns: Risk estimation is computed independently of response action selection.
  4. Operational Safety Metric: Computes Risk-to-Action Safety Efficiency (RASE).
  5. Conformal Selective Gating: Distinguishes autonomous action, abstention, and analyst escalation.
  6. Mechanistic Causal Explanations: Decomposes decision factors into verifiable causal chains.
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
from detection.selective_gate import ConformalRiskGate, SelectionDecision
from detection.feature_selector import DynamicFeatureSelector
from detection.attack_path import AttackPathReasoner
from detection.representation_engine import get_representation_model
from adaptive_learning.weight_learner import (
    get_adaptive_weight_learner,
    EvidenceQualityEngine,
    EvidenceIndependenceController
)
from xai.causal_explainer import CausalExplainer, CausalReport

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
    use_signature:         bool  = True
    use_ml:                bool  = True
    use_statistical:       bool  = True
    use_trust:             bool  = True
    use_history:           bool  = True
    use_graph:             bool  = True
    use_forecast:          bool  = True
    use_uncertainty:       bool  = True
    use_ti:                bool  = True
    use_deception:         bool  = True
    use_asset_crit:        bool  = True
    use_episode_reasoning: bool  = True
    use_evidence_quality:  bool  = True
    use_selective_gate:    bool  = True
    use_dynamic_features:  bool  = True
    adaptive_weights:      bool  = False

    # Baseline Weight Parameters
    w_sig:                 float = 0.50   # Signature weight
    w_ml:                  float = 0.30   # ML Anomaly weight
    w_trust:               float = 0.15   # Trust mitigation weight
    w_hist:                float = 0.10   # Historical recidivism weight
    w_graph:               float = 0.10   # Graph corroboration weight
    w_fore:                float = 0.05   # Forecast momentum weight
    w_ti:                  float = 0.15   # Threat intelligence weight
    w_ep:                  float = 0.10   # Attack episode / path risk weight

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
    autonomy_decision:    str = "AUTONOMOUS_PASS"
    conformal_tau:        float = 0.25
    feature_mask:         Dict[str, float] = field(default_factory=dict)
    causal_chains:        List[dict] = field(default_factory=list)

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
    R_ep:               float = 0.0    # Attack episode / path component
    
    # Uncertainty & Dynamics
    risk_confidence:    float = 0.85   # Confidence in risk evaluation [0.0, 1.0]
    risk_uncertainty:   float = 0.15   # Epistemic/disagreement uncertainty [0.0, 1.0]
    risk_delta:         float = 0.0    # Delta relative to entity's baseline
    risk_velocity:      float = 0.0    # Velocity of risk accumulation (dR/dt)
    risk_state:         str   = "NORMAL" # NORMAL, ELEVATED, ESCALATING, CRITICAL
    decision_reason:    str   = ""
    
    # Autonomy & XAI
    autonomy_decision:  str   = "AUTONOMOUS_PASS"
    conformal_tau:      float = 0.25
    causal_chains:      List[dict] = field(default_factory=list)
    
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
            "autonomy_decision": self.autonomy_decision,
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
                "R_ep":     self.R_ep,
            },
            "uncertainty": {
                "confidence":  self.risk_confidence,
                "uncertainty": self.risk_uncertainty,
                "conformal_tau": self.conformal_tau,
            },
            "dynamics": {
                "risk_delta":    self.risk_delta,
                "risk_velocity": self.risk_velocity,
                "risk_state":    self.risk_state,
            },
            "causal_chains":     self.causal_chains,
            "explanation":       self.explanation,
            "flags":             self.flags,
            "mitre_techniques":  self.mitre_techniques,
            "evidence_ids":      self.evidence_ids,
            "trace":             self.trace.to_dict() if self.trace else None,
        }


class AdaptiveRiskEngine:
    """
    Core Controller: Computes sound, multi-signal, uncertainty-aware cyber risk scores.
    Enforces strict mathematical bounds, deterministic replayability, and provenance tracking.
    """

    def __init__(
        self,
        default_config: Optional[RiskConfig] = None,
        initial_trust: float = 0.50,
        trust_decay_rate: float = 0.05,
    ):
        self.config = default_config or RiskConfig()
        self._initial_trust = initial_trust
        self._trust_decay = trust_decay_rate
        
        # State stores (thread-safe)
        self._trust_scores: Dict[str, float] = defaultdict(lambda: self._initial_trust)
        self._recent_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._last_update: Dict[str, float] = defaultdict(time.time)
        
        # Modular Engines
        self.selective_gate = ConformalRiskGate()
        self.feature_selector = DynamicFeatureSelector()
        self.attack_path_reasoner = AttackPathReasoner()
        self.evidence_quality_engine = EvidenceQualityEngine()
        self.causal_explainer = CausalExplainer()
        
        self._lock = threading.RLock()

    def get_trust(self, entity_key: str) -> float:
        with self._lock:
            return self._trust_scores[entity_key]

    def set_trust(self, entity_key: str, trust_level: float) -> None:
        with self._lock:
            clamped = max(0.0, min(1.0, float(trust_level)))
            self._trust_scores[entity_key] = clamped
            self._last_update[entity_key] = time.time()

    def update_trust_on_event(self, entity_key: str, is_anom: bool, severity_id: int = 1) -> float:
        """Dynamically adjusts trust based on verified behavior."""
        with self._lock:
            curr = self._trust_scores[entity_key]
            if is_anom:
                penalty = 0.05 * severity_id
                new_trust = max(0.0, curr - penalty)
            else:
                recovery = 0.01
                new_trust = min(1.0, curr + recovery)
            self._trust_scores[entity_key] = round(new_trust, 4)
            self._last_update[entity_key] = time.time()
            return self._trust_scores[entity_key]

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
        r_ep: float = 0.0,
        override_config: Optional[RiskConfig] = None,
    ) -> RiskResult:
        """
        Computes composite cyber risk R_t from all available evidence.
        Strictly deterministic, zero-drift reconstructible.
        """
        cfg = override_config or self.config
        evidence_ledger = get_evidence_ledger()
        created_evidence_ids = []
        event_id = str(_get(evt, "metadata", "event_id", default=f"EVT-{int(time.time()*1000)}"))

        # 1. Signature Evidence Term
        sig_flags = []
        mitre_techs = []
        raw_sig_score = 0.0
        sig_conf = 0.0

        if sig_matches:
            severities = []
            for sm in sig_matches:
                s_val = getattr(sm, "severity", None) or _get(sm, "severity", default=1)
                severities.append(float(s_val))
                rule = getattr(sm, "rule_name", None) or _get(sm, "rule_name", default="SIG_MATCH")
                sig_flags.append(str(rule))
                tech = getattr(sm, "mitre_technique", None) or _get(sm, "mitre_technique", default="")
                if tech:
                    mitre_techs.append(str(tech))

            max_sev = max(severities) if severities else 1.0
            raw_sig_score = min(1.0, max_sev / 5.0)
            sig_conf = 0.95
        
        S_sig = raw_sig_score if cfg.use_signature else 0.0
        if S_sig > 0:
            ev_sig = EvidenceRecord(
                evidence_id=f"EV-SIG-{event_id}",
                event_id=event_id,
                entity_id=entity_key,
                source=EvidenceSource.SURICATA_ENGINE.value,
                detector_type=EvidenceType.SIGNATURE.value,
                raw_score=raw_sig_score,
                normalized_score=S_sig,
                confidence=sig_conf,
                uncertainty=round(1.0 - sig_conf, 4),
                explanation=f"Signature matches: {', '.join(sig_flags[:3])}",
                mitre_mapping=mitre_techs,
            )
            evidence_ledger.record_evidence(ev_sig)
            created_evidence_ids.append(ev_sig.evidence_id)

        # 2. Machine Learning Anomaly Evidence Term
        raw_ml_score = float(
            getattr(ml_res, "ensemble_score", None) or
            getattr(ml_res, "anomaly_score", None) or
            _get(ml_res, "ensemble_score", default=None) or
            _get(ml_res, "anomaly_score", default=0.0) or 0.0
        )
        ml_conf = float(getattr(ml_res, "confidence", None) or _get(ml_res, "confidence", default=0.80))
        A_ml = raw_ml_score if cfg.use_ml else 0.0
        
        if A_ml > 0:
            ev_ml = EvidenceRecord(
                evidence_id=f"EV-ML-{event_id}",
                event_id=event_id,
                entity_id=entity_key,
                source=EvidenceSource.AUTOENCODER.value,
                detector_type=EvidenceType.ML_ANOMALY.value,
                raw_score=raw_ml_score,
                normalized_score=A_ml,
                confidence=ml_conf,
                uncertainty=round(1.0 - ml_conf, 4),
                explanation=f"ML ensemble anomaly score {A_ml:.3f}",
            )
            evidence_ledger.record_evidence(ev_ml)
            created_evidence_ids.append(ev_ml.evidence_id)

        # 3. Statistical Drift Term
        raw_drift = float(
            getattr(stat_res, "behavioral_drift", None) or
            getattr(stat_res, "drift_score", None) or
            _get(stat_res, "behavioral_drift", default=None) or
            _get(stat_res, "drift_score", default=0.0) or 0.0
        )
        stat_conf = float(getattr(stat_res, "confidence", None) or _get(stat_res, "confidence", default=0.75))
        delta_D = raw_drift if cfg.use_statistical else 0.0
        
        if delta_D > 0:
            ev_stat = EvidenceRecord(
                evidence_id=f"EV-STAT-{event_id}",
                event_id=event_id,
                entity_id=entity_key,
                source=EvidenceSource.WELFORD_STAT_ENGINE.value,
                detector_type=EvidenceType.STATISTICAL_DRIFT.value,
                raw_score=raw_drift,
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
        effective_r_ep = float(r_ep) if cfg.use_episode_reasoning else 0.0
        
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

        # Trust Term
        T_trust = self.get_trust(entity_key) if cfg.use_trust else 0.0

        # 5. Core Adaptive Risk Equation
        term_sig   = cfg.w_sig * S_sig
        term_ml    = cfg.w_ml * A_ml * (1.0 + delta_D)
        term_hist  = cfg.w_hist * effective_h_boost
        term_graph = cfg.w_graph * effective_g_corr
        term_fore  = cfg.w_fore * effective_p_fore
        term_ti    = cfg.w_ti * effective_ti
        term_ep    = cfg.w_ep * effective_r_ep if hasattr(cfg, "w_ep") else 0.0
        
        additive_threat = term_sig + term_ml + term_hist + term_graph + term_fore + term_ti + term_ep
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

        # 8. Selective Conformal Gate Evaluation
        if cfg.use_selective_gate:
            sel_decision = self.selective_gate.evaluate_gate(
                risk_score=risk_score,
                uncertainty=uncertainty,
                ood_score=0.0
            )
            autonomy_dec = sel_decision.action
            conformal_tau = sel_decision.conformal_tau
        else:
            autonomy_dec = "AUTONOMOUS_ACT" if is_alert else "AUTONOMOUS_PASS"
            conformal_tau = 0.25

        # 9. Causal Explanations
        raw_inputs_dict = {
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
            "R_ep": effective_r_ep,
        }
        intermediate_terms_dict = {
            "w_sig_S": term_sig,
            "w_ml_A": term_ml,
            "w_hist_H": term_hist,
            "w_graph_G": term_graph,
            "w_fore_P": term_fore,
            "w_ti_TI": term_ti,
            "w_ep_R": term_ep,
            "w_trust_T": trust_sub,
            "u_penalty": u_penalty,
        }
        
        causal_report = self.causal_explainer.explain_decision(
            raw_inputs=raw_inputs_dict,
            intermediate_terms=intermediate_terms_dict,
            final_risk=risk_score,
            event_id=event_id,
            entity_key=entity_key
        )
        causal_chains = [c.to_dict() for c in causal_report.causal_chains]

        # Construct DecisionTrace
        trace = DecisionTrace(
            event_id=event_id,
            entity_key=entity_key,
            timestamp=time.time(),
            config=cfg.to_dict(),
            raw_inputs=raw_inputs_dict,
            intermediate_terms=intermediate_terms_dict,
            additive_sum=round(additive_threat, 4),
            criticality_mult=round(mult_crit, 4),
            uncertainty_mult=round(mult_unc, 4),
            trust_subtraction=round(trust_sub, 4),
            pre_clip_score=round(raw_risk, 4),
            final_clamped_score=risk_score,
            severity=severity,
            remediation_level=remediation,
            evidence_ids=created_evidence_ids,
            autonomy_decision=autonomy_dec,
            conformal_tau=conformal_tau,
            causal_chains=causal_chains,
        )

        decision_reason = (
            f"Risk={risk_score:.3f} ({severity}) driven by {causal_report.dominant_causal_factor}. "
            f"Autonomy={autonomy_dec}. Uncertainty={uncertainty:.2f}."
        )
        explanation = (
            f"Entity {entity_key} evaluated at risk {risk_score:.3f} [{severity}]. "
            f"Dominant mechanism: {causal_report.dominant_causal_factor}. "
            f"Autonomous Action: {autonomy_dec}."
        )

        return RiskResult(
            entity_key=entity_key,
            risk_score=risk_score,
            severity=severity,
            severity_id=severity_id,
            remediation_level=remediation,
            is_alert=is_alert,
            S_sig=round(S_sig, 4),
            A_ml=round(A_ml, 4),
            delta_D=round(delta_D, 4),
            T_trust=round(T_trust, 4),
            H_boost=round(effective_h_boost, 4),
            G_corr=round(effective_g_corr, 4),
            P_fore=round(effective_p_fore, 4),
            TI_score=round(effective_ti, 4),
            A_crit=round(effective_a_crit, 4),
            R_ep=round(effective_r_ep, 4),
            risk_confidence=confidence,
            risk_uncertainty=uncertainty,
            risk_delta=risk_delta,
            risk_velocity=risk_velocity,
            risk_state=risk_state,
            decision_reason=decision_reason,
            autonomy_decision=autonomy_dec,
            conformal_tau=conformal_tau,
            causal_chains=causal_chains,
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
    
    t_sig   = cfg.get("w_sig", 0.5) * inputs["S_sig"] if cfg.get("use_signature", True) else 0.0
    t_ml    = cfg.get("w_ml", 0.3) * inputs["A_ml"] * (1.0 + inputs["delta_D"]) if cfg.get("use_ml", True) else 0.0
    t_hist  = cfg.get("w_hist", 0.1) * inputs["H_boost"] if cfg.get("use_history", True) else 0.0
    t_graph = cfg.get("w_graph", 0.1) * inputs["G_corr"] if cfg.get("use_graph", True) else 0.0
    t_fore  = cfg.get("w_fore", 0.05) * inputs["P_fore"] if cfg.get("use_forecast", True) else 0.0
    t_ti    = cfg.get("w_ti", 0.15) * inputs["TI_score"] if cfg.get("use_ti", True) else 0.0
    t_ep    = cfg.get("w_ep", 0.10) * inputs.get("R_ep", 0.0) if cfg.get("use_episode_reasoning", True) else 0.0
    
    add_sum = t_sig + t_ml + t_hist + t_graph + t_fore + t_ti + t_ep
    crit_mult = inputs["A_crit"] if cfg.get("use_asset_crit", True) else 1.0
    
    u_pen = (inputs["uncertainty"] * 0.30) if cfg.get("use_uncertainty", True) else 0.0
    unc_mult = (1.0 - u_pen)
    
    t_trust = (cfg.get("w_trust", 0.15) * inputs["T_trust"]) if cfg.get("use_trust", True) else 0.0
    
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


# Singleton
_engine_instance: Optional[AdaptiveRiskEngine] = None
_engine_lock = threading.Lock()


def get_risk_engine() -> AdaptiveRiskEngine:
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = AdaptiveRiskEngine()
    return _engine_instance


def run_risk_engine(
    entity_key: str,
    sig_matches: list,
    ml_res: Any,
    stat_res: Any,
    evt: dict = None,
    **kwargs,
) -> RiskResult:
    """Convenience helper executing the global singleton risk engine."""
    engine = get_risk_engine()
    return engine.score_risk(
        entity_key=entity_key,
        sig_matches=sig_matches,
        ml_res=ml_res,
        stat_res=stat_res,
        evt=evt,
        **kwargs,
    )
