"""
AHRAS Conformal Response Controller
-----------------------------------
Implements split conformal prediction with finite-sample coverage guarantees
for autonomous response actions, eliminating false containment and unwarranted autonomy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ConformalSafetyDecision:
    """
    Result of conformal prediction set evaluation for response gating.
    """
    prediction_set: List[int]       # Subset of {0, 1}
    is_singleton: bool
    is_autonomous_candidate: bool
    conformal_tau: float            # Calibrated nonconformity quantile
    attack_nonconformity: float     # 1 - P(Attack)
    benign_nonconformity: float     # 1 - P(Benign)
    epistemic_uncertainty: float
    gating_recommendation: str      # e.g. AUTONOMOUS_CONTAINMENT, AUTONOMOUS_PASS, ABSTAIN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_set": self.prediction_set,
            "is_singleton": self.is_singleton,
            "is_autonomous_candidate": self.is_autonomous_candidate,
            "conformal_tau": round(self.conformal_tau, 4),
            "attack_nonconformity": round(self.attack_nonconformity, 4),
            "benign_nonconformity": round(self.benign_nonconformity, 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "gating_recommendation": self.gating_recommendation,
        }


class ConformalResponseController:
    """
    Calibrates nonconformity quantiles and gates automated containment.
    """
    def __init__(
        self,
        alpha: float = 0.05,            # 95% guaranteed coverage boundary
        max_uncertainty_tau: float = 0.25,
    ) -> None:
        self.alpha = alpha
        self.max_uncertainty_tau = max_uncertainty_tau
        self.conformal_tau: float = 0.50
        self.is_calibrated: bool = False

    def calibrate(
        self,
        probabilities: np.ndarray,      # P(Y = 1 | x) for calibration instances
        ground_truth: np.ndarray,        # True labels in {0, 1}
    ) -> float:
        """
        Computes finite-sample nonconformity quantile threshold:
        tau = Quantile(s_i, ceil((n + 1) * (1 - alpha)) / n)
        """
        n = len(ground_truth)
        scores = []
        for p, y in zip(probabilities, ground_truth):
            prob_true = p if y == 1 else (1.0 - p)
            s_i = 1.0 - float(prob_true)
            scores.append(s_i)

        scores_sorted = np.sort(scores)
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1
        q_idx = min(max(0, q_idx), n - 1)
        self.conformal_tau = float(scores_sorted[q_idx])
        self.is_calibrated = True
        return self.conformal_tau

    def evaluate_instance(
        self,
        p_attack: float,
        epistemic_uncertainty: float = 0.05,
    ) -> ConformalSafetyDecision:
        """
        Forms conformal prediction set C(x) and determines gating action.
        """
        s_attack = 1.0 - p_attack
        s_benign = p_attack

        pred_set = []
        if s_benign <= self.conformal_tau:
            pred_set.append(0)
        if s_attack <= self.conformal_tau:
            pred_set.append(1)

        is_singleton = len(pred_set) == 1
        is_autonomous = False
        action = "ABSTAIN"

        if is_singleton and epistemic_uncertainty <= self.max_uncertainty_tau:
            if pred_set == [1]:
                # Provable attack with high confidence
                is_autonomous = True
                action = "AUTONOMOUS_CONTAINMENT"
            elif pred_set == [0]:
                # Provable benign
                is_autonomous = True
                action = "AUTONOMOUS_PASS"
        else:
            if len(pred_set) > 1:
                # Ambiguous: Both benign and attack are plausible under conformal boundary
                action = "ABSTAIN"
            elif len(pred_set) == 0:
                # Highly anomalous / OOD
                action = "ESCALATE_ANALYST"
            elif epistemic_uncertainty > self.max_uncertainty_tau:
                action = "ESCALATE_ANALYST"

        return ConformalSafetyDecision(
            prediction_set=pred_set,
            is_singleton=is_singleton,
            is_autonomous_candidate=is_autonomous,
            conformal_tau=self.conformal_tau,
            attack_nonconformity=s_attack,
            benign_nonconformity=s_benign,
            epistemic_uncertainty=epistemic_uncertainty,
            gating_recommendation=action,
        )
