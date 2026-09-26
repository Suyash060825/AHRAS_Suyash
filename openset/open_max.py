"""
AHRAS OpenMax Extreme Value Classifier
--------------------------------------
Implements OpenMax calibrated activation re-weighting via Weibull tail modeling
(Bendale & Boult, CVPR 2016) to detect unknown/zero-day attacks without retraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import weibull_min


@dataclass
class OpenMaxOutput:
    """
    Open-world classification result with an explicit unknown class.
    """
    predicted_class: int            # 0..K-1 if known, K if UNKNOWN_ZERO_DAY
    is_unknown: bool
    openmax_probabilities: np.ndarray  # (K+1)-dimensional probability vector
    unknown_probability: float
    max_known_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": int(self.predicted_class),
            "is_unknown": self.is_unknown,
            "unknown_probability": round(float(self.unknown_probability), 4),
            "max_known_probability": round(float(self.max_known_probability), 4),
        }


class OpenMaxEngine:
    """
    Fits class mean activation vectors (MAVs) and Weibull tail distributions on training logits.
    Recalibrates test activations into an explicit (K+1)-th Unknown class.
    """
    def __init__(self, tail_size: int = 20, alpha_rank: int = 3) -> None:
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank
        self.mavs: Dict[int, np.ndarray] = {}
        self.weibull_models: Dict[int, Any] = {}
        self.n_classes: int = 0

    def fit(self, activations: np.ndarray, labels: np.ndarray) -> None:
        """
        Calculates MAVs and fits Weibull models for each known class.
        activations: (N, K) array of penultimate logits/activations
        labels: (N,) true integer class labels
        """
        classes = np.unique(labels)
        self.n_classes = len(classes)

        for c in classes:
            idx = np.where(labels == c)[0]
            class_acts = activations[idx]
            mav = np.mean(class_acts, axis=0)
            self.mavs[int(c)] = mav

            # Compute Euclidean distances from MAV
            dists = np.linalg.norm(class_acts - mav, axis=1)
            # Take the largest distances (tail)
            tail_dists = np.sort(dists)[-min(len(dists), self.tail_size):]

            # Fit Weibull distribution on tail
            # scipy.stats.weibull_min fit
            try:
                # Add tiny jitter if all distances identical to prevent singular fit
                if np.all(tail_dists == tail_dists[0]):
                    tail_dists = tail_dists + np.random.normal(0, 1e-5, len(tail_dists))
                shape, loc, scale = weibull_min.fit(tail_dists, floc=0.0)
                self.weibull_models[int(c)] = (shape, loc, max(1e-5, scale))
            except Exception:
                # Fallback parameters
                self.weibull_models[int(c)] = (1.5, 0.0, max(1.0, float(np.mean(tail_dists))))

    def predict(self, activation_vector: np.ndarray, unknown_threshold: float = 0.50) -> OpenMaxOutput:
        """
        Calculates OpenMax recalibrated probabilities including class K (UNKNOWN).
        """
        K = self.n_classes
        if K == 0 or not self.mavs:
            # Fallback uncalibrated
            return OpenMaxOutput(0, False, np.array([1.0, 0.0]), 0.0, 1.0)

        # Ranked classes by activation
        ranked_classes = np.argsort(-activation_vector)
        modified_acts = activation_vector.copy()
        unknown_act = 0.0

        for r in range(min(self.alpha_rank, K)):
            c = int(ranked_classes[r])
            if c not in self.mavs or c not in self.weibull_models:
                continue

            mav = self.mavs[c]
            dist = np.linalg.norm(activation_vector - mav)
            shape, loc, scale = self.weibull_models[c]

            # Cumulative distribution function of Weibull
            cdf = weibull_min.cdf(dist, shape, loc=loc, scale=scale)
            # Alpha weight decay based on rank: alpha_w(r) = (alpha - r) / alpha
            alpha_w = (self.alpha_rank - r) / float(max(1, self.alpha_rank))
            w_c = 1.0 - (cdf * alpha_w)

            # Re-weight activation
            modified_acts[c] = activation_vector[c] * w_c
            unknown_act += activation_vector[c] * (1.0 - w_c)

        # Concatenate unknown activation to form (K+1)-dim vector
        all_acts = np.append(modified_acts, max(0.0, unknown_act))
        # Softmax with numerical stability
        exp_acts = np.exp(all_acts - np.max(all_acts))
        openmax_probs = exp_acts / np.sum(exp_acts)

        p_unknown = float(openmax_probs[-1])
        known_probs = openmax_probs[:-1]
        best_known_class = int(np.argmax(known_probs))
        p_known_max = float(known_probs[best_known_class])

        is_unknown = p_unknown >= unknown_threshold
        pred_class = K if is_unknown else best_known_class

        return OpenMaxOutput(
            predicted_class=pred_class,
            is_unknown=is_unknown,
            openmax_probabilities=openmax_probs,
            unknown_probability=p_unknown,
            max_known_probability=p_known_max,
        )
