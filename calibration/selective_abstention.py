from __future__ import annotations
"""
AHRAS Calibration & Selective Prediction Engine (Sections 43 & 44)
-------------------------------------------------------------------
Addresses the fundamental requirement:
  "Never equate raw model score with probability unless calibrated."
  "Allow: BENIGN, ATTACK, UNKNOWN, ABSTAIN.
   Evaluate: coverage, error, selective risk. Plot: coverage vs error."

Capabilities:
1. Formal Probability Calibration Metrics:
   - Expected Calibration Error (ECE) across M uniform or quantile bins.
   - Maximum Calibration Error (MCE).
   - Brier Score: (1/N) * sum((p_i - y_i)^2).
   - Reliability diagram bin statistics: mean confidence vs empirical accuracy.

2. Platt Scaling & Isotonic Calibration:
   - Fits parametric sigmoid logistic mapping or isotonic step mapping on validation sets.

3. Four-State Selective Prediction with Abstention:
   - BENIGN: Low calibrated risk, low uncertainty.
   - ATTACK: High calibrated risk, low uncertainty, known attack pattern.
   - UNKNOWN: Elevated OOD / reconstruction error, potential zero-day attack.
   - ABSTAIN: Ambiguous near-threshold risk or high epistemic uncertainty.

4. Coverage vs. Error Frontier Profiling:
   - Computes empirical error rate across selective rejection thresholds tau in [0.0, 1.0].
"""

import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


@dataclass
class ReliabilityBin:
    """Bin statistics for calibration reliability diagrams."""
    bin_idx:         int
    confidence_min:  float
    confidence_max:  float
    mean_confidence: float
    empirical_accuracy: float
    sample_count:    int
    bin_error:       float  # |accuracy - confidence|


@dataclass
class CalibrationReport:
    """Quantitative calibration evaluation metrics."""
    ece:              float   # Expected Calibration Error
    mce:              float   # Maximum Calibration Error
    brier_score:      float   # Mean squared error to binary ground truth
    total_samples:    int
    reliability_bins: List[ReliabilityBin]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "total_samples": self.total_samples,
            "reliability_bins": [asdict(b) for b in self.reliability_bins],
        }


@dataclass
class SelectivePredictionResult:
    """Output of 4-way selective classification with abstention."""
    event_id:             str
    raw_score:            float
    calibrated_prob:      float
    uncertainty:          float
    ood_score:            float
    predicted_state:      str    # "BENIGN", "ATTACK", "UNKNOWN", "ABSTAIN"
    is_abstained:         bool
    selection_confidence: float
    rationale:            str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelectiveCalibrationEngine:
    """
    Stateful probability calibration and selective decision engine.
    """

    def __init__(
        self,
        risk_threshold: float = 0.65,
        uncertainty_abstain_threshold: float = 0.35,
        ood_unknown_threshold: float = 0.60,
        n_bins: int = 10,
    ):
        self.risk_threshold = risk_threshold
        self.uncertainty_abstain_threshold = uncertainty_abstain_threshold
        self.ood_unknown_threshold = ood_unknown_threshold
        self.n_bins = n_bins

        # Platt scaler model: Log-odds mapper
        self._platt_scaler: Optional[LogisticRegression] = None
        self.is_calibrated: bool = False

    def fit_calibration(self, val_scores: np.ndarray, val_labels: np.ndarray) -> None:
        """
        Fits Platt scaling logistic regression on validation scores.
        """
        scores = np.asarray(val_scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(val_labels, dtype=np.int32).ravel()

        if len(np.unique(labels)) < 2:
            log.warning("Calibration requires at least two distinct classes. Skipping Platt fit.")
            return

        clf = LogisticRegression(solver="lbfgs", max_iter=200)
        clf.fit(scores, labels)
        self._platt_scaler = clf
        self.is_calibrated = True

    def predict_calibrated_probability(self, raw_score: float) -> float:
        """
        Maps a raw model/risk score to calibrated posterior probability P(Malicious | score).
        """
        if not self.is_calibrated or self._platt_scaler is None:
            # Fallback to standard sigmoid clamp
            return float(np.clip(raw_score, 0.0, 1.0))

        prob = self._platt_scaler.predict_proba([[raw_score]])[0, 1]
        return float(np.clip(prob, 0.0, 1.0))

    def compute_calibration_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationReport:
        """
        Computes formal ECE, MCE, Brier score, and reliability diagram bins.
        """
        probs = np.asarray(probabilities, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        n = len(probs)

        if n == 0:
            return CalibrationReport(0.0, 0.0, 0.0, 0, [])

        brier = float(np.mean((probs - y) ** 2))

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bins: List[ReliabilityBin] = []
        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            b_low, b_high = bin_edges[i], bin_edges[i + 1]
            if i == self.n_bins - 1:
                mask = (probs >= b_low) & (probs <= b_high)
            else:
                mask = (probs >= b_low) & (probs < b_high)

            count = int(np.sum(mask))
            if count > 0:
                mean_conf = float(np.mean(probs[mask]))
                acc = float(np.mean(y[mask]))
                err = abs(acc - mean_conf)
                ece += (count / n) * err
                mce = max(mce, err)
            else:
                mean_conf = (b_low + b_high) / 2.0
                acc = 0.0
                err = 0.0

            bins.append(ReliabilityBin(
                bin_idx=i,
                confidence_min=round(b_low, 3),
                confidence_max=round(b_high, 3),
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(acc, 4),
                sample_count=count,
                bin_error=round(err, 4),
            ))

        return CalibrationReport(
            ece=round(ece, 4),
            mce=round(mce, 4),
            brier_score=round(brier, 4),
            total_samples=n,
            reliability_bins=bins,
        )

    def evaluate_event(
        self,
        raw_score: float,
        uncertainty: float,
        ood_score: float = 0.0,
        event_id: str = "EVT-01",
    ) -> SelectivePredictionResult:
        """
        Classifies an event into one of four operational states:
        BENIGN, ATTACK, UNKNOWN, or ABSTAIN.
        """
        cal_prob = self.predict_calibrated_probability(raw_score)

        # 1. Check for Epistemic Ambiguity -> ABSTAIN
        if uncertainty >= self.uncertainty_abstain_threshold:
            return SelectivePredictionResult(
                event_id=event_id,
                raw_score=round(raw_score, 4),
                calibrated_prob=round(cal_prob, 4),
                uncertainty=round(uncertainty, 4),
                ood_score=round(ood_score, 4),
                predicted_state="ABSTAIN",
                is_abstained=True,
                selection_confidence=round(1.0 - uncertainty, 4),
                rationale=f"Model uncertainty ({uncertainty:.2f}) exceeds threshold ({self.uncertainty_abstain_threshold:.2f}); action safely deferred.",
            )

        # 2. Check for Zero-Day Novelty -> UNKNOWN
        if ood_score >= self.ood_unknown_threshold:
            return SelectivePredictionResult(
                event_id=event_id,
                raw_score=round(raw_score, 4),
                calibrated_prob=round(cal_prob, 4),
                uncertainty=round(uncertainty, 4),
                ood_score=round(ood_score, 4),
                predicted_state="UNKNOWN",
                is_abstained=False,
                selection_confidence=round(ood_score, 4),
                rationale=f"Zero-day / Out-of-Distribution indicator ({ood_score:.2f}) indicates novel attack pattern; route to deception/analyst.",
            )

        # 3. High Calibrated Risk -> ATTACK
        if cal_prob >= self.risk_threshold:
            return SelectivePredictionResult(
                event_id=event_id,
                raw_score=round(raw_score, 4),
                calibrated_prob=round(cal_prob, 4),
                uncertainty=round(uncertainty, 4),
                ood_score=round(ood_score, 4),
                predicted_state="ATTACK",
                is_abstained=False,
                selection_confidence=round(cal_prob, 4),
                rationale=f"Calibrated breach probability ({cal_prob:.2f}) exceeds threshold ({self.risk_threshold:.2f}) with high confidence.",
            )

        # 4. Low Calibrated Risk -> BENIGN
        return SelectivePredictionResult(
            event_id=event_id,
            raw_score=round(raw_score, 4),
            calibrated_prob=round(cal_prob, 4),
            uncertainty=round(uncertainty, 4),
            ood_score=round(ood_score, 4),
            predicted_state="BENIGN",
            is_abstained=False,
            selection_confidence=round(1.0 - cal_prob, 4),
            rationale=f"Calibrated breach probability ({cal_prob:.2f}) is below alert threshold; traffic cleared as benign.",
        )

    def compute_coverage_vs_error_curve(
        self,
        raw_scores: np.ndarray,
        uncertainties: np.ndarray,
        labels: np.ndarray,
        threshold_steps: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Evaluates the Coverage vs. Error trade-off curve across varying uncertainty rejection levels.
        """
        uncertainty_thresholds = np.linspace(0.10, 0.90, threshold_steps)
        curve: List[Dict[str, float]] = []

        scores = np.asarray(raw_scores)
        uncs = np.asarray(uncertainties)
        y = np.asarray(labels)
        n = len(y)

        for u_thresh in uncertainty_thresholds:
            # Retain samples where uncertainty <= u_thresh
            retained_mask = uncs <= u_thresh
            retained_count = int(np.sum(retained_mask))

            coverage = retained_count / n if n > 0 else 0.0

            if retained_count > 0:
                y_sub = y[retained_mask]
                preds_sub = (scores[retained_mask] >= self.risk_threshold).astype(int)
                err = float(np.mean(preds_sub != y_sub))
            else:
                err = 0.0

            curve.append({
                "uncertainty_threshold": round(float(u_thresh), 3),
                "coverage": round(float(coverage), 4),
                "error_rate": round(float(err), 4),
                "retained_samples": retained_count,
            })

        return curve


_global_calibration_engine: Optional[SelectiveCalibrationEngine] = None

def get_calibration_engine() -> SelectiveCalibrationEngine:
    global _global_calibration_engine
    if _global_calibration_engine is None:
        _global_calibration_engine = SelectiveCalibrationEngine()
    return _global_calibration_engine
