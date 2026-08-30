from __future__ import annotations
"""
AHRAS Module — Conformal & Selective Risk Gating Engine
-------------------------------------------------------
Implements statistically sound selective risk decisioning using split conformal prediction
and calibrated nonconformity scoring:

  1. Calibration Phase:
       Computes nonconformity scores q_i = |y_i - R_i| on hold-out validation events.
       Calculates conformal quantile threshold tau* = Quantile_{1 - alpha}(q) to guarantee
       marginal coverage >= 1 - alpha on exchangeable test distributions.

  2. Selective Inference:
       Distinguishes:
         - "AUTONOMOUS": High confidence, low epistemic uncertainty, risk firmly separated from threshold.
         - "ABSTAIN": Ambiguous risk, high disagreement uncertainty, or conformal nonconformity exceeds safety band.
         - "ESCALATE_ANALYST": Critical risk under uncertainty; triggers human-in-the-loop triage.
"""

import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SelectionDecision:
    """Represents a selective prediction / gating decision."""
    action:             str      # "AUTONOMOUS_ACT", "AUTONOMOUS_PASS", "ABSTAIN", "ESCALATE_ANALYST"
    target_coverage:    float    # e.g., 0.90
    conformal_tau:      float    # Calibrated nonconformity quantile threshold
    nonconformity_score: float   # Nonconformity score for this instance
    uncertainty_level:  float    # Epistemic/model uncertainty
    abstain_reason:     str      # Explanation if action is ABSTAIN or ESCALATE
    is_autonomous:      bool     # True if safe for automated closed-loop execution

    def to_dict(self) -> dict:
        return asdict(self)


class ConformalRiskGate:
    """
    Split Conformal Risk Gate for uncertainty-aware selective risk execution.
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
        self.calibrated_tau: float = 0.25  # Default empirical tau before calibration
        self.is_calibrated: bool = False
        self._calibration_scores: List[float] = []

    def calibrate(self, risk_scores: List[float], labels: List[int]) -> float:
        """
        Calibrate conformal nonconformity threshold tau* on validation dataset.
        risk_scores: array of predicted risk in [0, 1]
        labels: binary labels (0 = benign, 1 = attack)
        """
        if len(risk_scores) == 0 or len(risk_scores) != len(labels):
            raise ValueError("Calibration requires non-empty matching risk scores and labels.")

        nonconformity = np.abs(np.array(labels, dtype=np.float64) - np.array(risk_scores, dtype=np.float64))
        n = len(nonconformity)
        
        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.calibrated_tau = float(np.quantile(nonconformity, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
        self.calibrated_tau = max(0.01, min(1.0, self.calibrated_tau))
        self.is_calibrated = True
        self._calibration_scores = list(nonconformity)
        
        log.info(f"ConformalRiskGate calibrated on n={n} samples: tau*={self.calibrated_tau:.4f} @ coverage={self.target_coverage:.2f}")
        return self.calibrated_tau

    def evaluate_gate(
        self,
        risk_score: float,
        uncertainty: float,
        ood_score: float = 0.0
    ) -> SelectionDecision:
        """
        Evaluates whether a risk evaluation should trigger autonomous action, abstention, or analyst escalation.
        """
        # Estimated nonconformity against the binary decision boundary
        # If risk is near boundary (0.5), nonconformity is highest
        pseudo_label = 1.0 if risk_score >= self.risk_action_threshold else 0.0
        instance_nonconf = abs(risk_score - pseudo_label)
        
        reasons = []
        is_ambiguous = (abs(risk_score - self.risk_action_threshold) < 0.15)
        is_high_uncertainty = (uncertainty >= self.uncertainty_threshold)
        is_ood = (ood_score >= 0.65)
        
        if is_high_uncertainty:
            reasons.append(f"Model uncertainty ({uncertainty:.3f}) >= threshold ({self.uncertainty_threshold:.3f})")
        if is_ood:
            reasons.append(f"OOD score ({ood_score:.3f}) suggests novel/unseen behavior pattern")
        if instance_nonconf > self.calibrated_tau and is_ambiguous:
            reasons.append(f"Nonconformity ({instance_nonconf:.3f}) exceeds conformal band tau* ({self.calibrated_tau:.3f})")

        if is_high_uncertainty or is_ood or (len(reasons) > 0 and is_ambiguous):
            if risk_score >= self.risk_action_threshold:
                action = "ESCALATE_ANALYST"
                abstain_msg = f"High-risk alert requires analyst verification: {'; '.join(reasons)}"
            else:
                action = "ABSTAIN"
                abstain_msg = f"Abstained from autonomous decision due to: {'; '.join(reasons)}"
            is_auto = False
        else:
            if risk_score >= self.risk_action_threshold:
                action = "AUTONOMOUS_ACT"
                abstain_msg = "Safe autonomous containment authorized within calibrated conformal bounds."
            else:
                action = "AUTONOMOUS_PASS"
                abstain_msg = "Safe autonomous pass authorized within calibrated conformal bounds."
            is_auto = True

        return SelectionDecision(
            action=action,
            target_coverage=self.target_coverage,
            conformal_tau=self.calibrated_tau,
            nonconformity_score=round(instance_nonconf, 4),
            uncertainty_level=round(uncertainty, 4),
            abstain_reason=abstain_msg,
            is_autonomous=is_auto
        )
