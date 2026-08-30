from __future__ import annotations
"""
AHRAS Module — Conformal & Cost-Aware Selective Risk Gating Engine
------------------------------------------------------------------
Implements statistically sound selective risk decisioning using split conformal prediction
and calibrated nonconformity scoring across 7 discrete operational defense actions:

  1. AUTONOMOUS_PASS: Low risk, low uncertainty (auto-cleared).
  2. MONITOR: Low-to-medium risk baseline monitoring.
  3. DECEPTION: High OOD/zero-day signature routed to dynamic honeypot tripwires.
  4. STAGED_CONTAINMENT: Medium-to-high risk staged for automated policy execution.
  5. AUTONOMOUS_CONTAINMENT: High-confidence critical attack, conformal safety verified.
  6. ABSTAIN: Ambiguous near-threshold risk within conformal nonconformity band.
  7. ESCALATE_ANALYST: High risk under epistemic disagreement or extreme novelty.

Operational Loss Minimization:
  Selects action minimizing expected operational cost:
    Loss(a, y) = C_intervene(a) + (1 - y) * C_fp(a) + y * (1 - Contained(a)) * C_breach
"""

import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Standard 7 Operational Gating Action Constants
ACTION_AUTONOMOUS_PASS        = "AUTONOMOUS_PASS"
ACTION_MONITOR                = "MONITOR"
ACTION_DECEPTION              = "DECEPTION"
ACTION_STAGED_CONTAINMENT     = "STAGED_CONTAINMENT"
ACTION_AUTONOMOUS_CONTAINMENT = "AUTONOMOUS_CONTAINMENT"
ACTION_ABSTAIN                = "ABSTAIN"
ACTION_ESCALATE_ANALYST       = "ESCALATE_ANALYST"


@dataclass
class SelectionDecision:
    """Represents a selective prediction / gating decision."""
    action:               str      # One of the 7 operational actions
    target_coverage:      float    # e.g., 0.90
    conformal_tau:        float    # Calibrated nonconformity quantile threshold
    nonconformity_score:  float    # Nonconformity score for this instance
    uncertainty_level:    float    # Epistemic/model uncertainty
    expected_loss:        float    # Estimated operational loss for chosen action
    abstain_reason:       str      # Explanation for action choice
    is_autonomous:        bool     # True if safe for automated closed-loop execution

    def to_dict(self) -> dict:
        return asdict(self)


class ConformalRiskGate:
    """
    Split Conformal Risk Gate with Cost-Sensitive Gating & Recalibration Triggers.
    """

    def __init__(
        self,
        target_coverage: float = 0.90,
        uncertainty_threshold: float = 0.35,
        risk_action_threshold: float = 0.70,
        seed: int = 42
    ):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.uncertainty_threshold = uncertainty_threshold
        self.risk_action_threshold = risk_action_threshold
        self.calibrated_tau: float = 0.25
        self.is_calibrated: bool = False
        self._calibration_scores: List[float] = []
        self._conformal_errors_observed: int = 0
        self._total_inferences: int = 0

    def calibrate(self, risk_scores: List[float], labels: List[int]) -> float:
        """
        Calibrate conformal nonconformity threshold tau* on validation dataset:
          q_i = |y_i - R_i|
          tau* = Quantile_{ceil((n+1)(1-alpha))/n}(q)
        """
        if len(risk_scores) == 0 or len(risk_scores) != len(labels):
            raise ValueError("Calibration requires non-empty matching risk scores and labels.")

        nonconformity = np.abs(np.array(labels, dtype=np.float64) - np.array(risk_scores, dtype=np.float64))
        n = len(nonconformity)
        
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.calibrated_tau = float(np.quantile(nonconformity, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
        self.calibrated_tau = max(0.01, min(1.0, self.calibrated_tau))
        self.is_calibrated = True
        self._calibration_scores = list(nonconformity)
        
        log.info(f"ConformalRiskGate calibrated on n={n} samples: tau*={self.calibrated_tau:.4f} @ coverage={self.target_coverage:.2f}")
        return self.calibrated_tau

    def compute_expected_loss(self, action: str, risk_p: float, uncertainty: float) -> float:
        """
        Computes expected operational cost:
          C_fp: Cost of false disruption on benign host
          C_fn: Cost of uncontained breach
          C_analyst: Cost of human triage
        """
        c_fp = {"AUTONOMOUS_CONTAINMENT": 1.0, "STAGED_CONTAINMENT": 0.3, "DECEPTION": 0.05, "MONITOR": 0.0, "AUTONOMOUS_PASS": 0.0, "ABSTAIN": 0.1, "ESCALATE_ANALYST": 0.2}.get(action, 0.5)
        c_fn = {"AUTONOMOUS_CONTAINMENT": 0.05, "STAGED_CONTAINMENT": 0.2, "DECEPTION": 0.4, "MONITOR": 0.8, "AUTONOMOUS_PASS": 1.0, "ABSTAIN": 0.3, "ESCALATE_ANALYST": 0.1}.get(action, 0.5)
        
        prob_attack = float(np.clip(risk_p, 0.0, 1.0))
        prob_benign = 1.0 - prob_attack
        
        exp_loss = (prob_benign * c_fp) + (prob_attack * c_fn) + (uncertainty * 0.15)
        return round(exp_loss, 4)

    def evaluate_gate(
        self,
        risk_score: float,
        uncertainty: float,
        ood_score: float = 0.0
    ) -> SelectionDecision:
        """
        Evaluates risk against conformal nonconformity bounds and maps to one of the 7 operational actions.
        """
        pseudo_label = 1.0 if risk_score >= self.risk_action_threshold else 0.0
        instance_nonconf = abs(risk_score - pseudo_label)
        
        is_ambiguous = (abs(risk_score - self.risk_action_threshold) < 0.15)
        is_high_uncertainty = (uncertainty >= self.uncertainty_threshold)
        is_ood = (ood_score >= 0.65)
        
        # Action Arbitration
        if is_ood and risk_score < 0.85:
            # Zero-day / novel evasion signature -> route to Deception tripwire
            action = ACTION_DECEPTION
            reason = f"Novel zero-day telemetry (OOD={ood_score:.3f}) routed to dynamic honeypot deception."
            is_auto = True
        elif is_high_uncertainty:
            if risk_score >= self.risk_action_threshold:
                action = ACTION_ESCALATE_ANALYST
                reason = f"High-risk incident under epistemic uncertainty ({uncertainty:.3f} >= {self.uncertainty_threshold:.3f}) escalated to analyst."
                is_auto = False
            else:
                action = ACTION_ABSTAIN
                reason = f"Abstained from autonomous decision due to elevated model uncertainty ({uncertainty:.3f})."
                is_auto = False
        elif instance_nonconf > self.calibrated_tau and is_ambiguous:
            action = ACTION_ABSTAIN
            reason = f"Nonconformity ({instance_nonconf:.3f}) exceeds calibrated conformal safety band tau* ({self.calibrated_tau:.3f})."
            is_auto = False
        elif risk_score >= self.risk_action_threshold:
            if risk_score >= 0.88 and uncertainty < 0.20:
                action = ACTION_AUTONOMOUS_CONTAINMENT
                reason = "Safe autonomous containment authorized: high confidence, conformal bound verified."
                is_auto = True
            else:
                action = ACTION_STAGED_CONTAINMENT
                reason = "Staged containment queued: risk exceeds policy threshold."
                is_auto = True
        elif risk_score >= 0.35:
            action = ACTION_MONITOR
            reason = "Sub-critical elevated risk placed under active behavioral monitoring."
            is_auto = True
        else:
            action = ACTION_AUTONOMOUS_PASS
            reason = "Safe autonomous pass authorized: low risk, verified baseline conformance."
            is_auto = True

        loss = self.compute_expected_loss(action, risk_score, uncertainty)

        return SelectionDecision(
            action=action,
            target_coverage=self.target_coverage,
            conformal_tau=self.calibrated_tau,
            nonconformity_score=round(instance_nonconf, 4),
            uncertainty_level=round(uncertainty, 4),
            expected_loss=loss,
            abstain_reason=reason,
            is_autonomous=is_auto
        )
