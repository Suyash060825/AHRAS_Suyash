from __future__ import annotations
"""
AHRAS Adaptive Weight Learning Engine
---------------------------------------
Enables online dynamic adaptation of risk fusion weights based on analyst
feedback (confirmed attack vs confirmed false positive).

Uses stochastic gradient descent on cross-entropy loss with simplex projection
to keep weights positive, bounded, and normalized.
"""

import math
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)

# Default weights
DEFAULT_WEIGHTS = {
    "signature":  0.40,
    "anomaly":    0.30,
    "density":    0.20,
    "drift_rate": 0.10,
}

MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.70
LEARNING_RATE = 0.02


@dataclass
class FeedbackSample:
    src_ip:         str
    label:          int                   # 1 = true attack, 0 = false positive / benign
    components:     Dict[str, float]      # component values in [0, 1]
    predicted_risk: float                 # model's risk score
    timestamp:      float = 0.0


class AdaptiveWeightLearner:
    """
    Online learner that tunes detector and risk component weights from SOC feedback.
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None, lr: float = LEARNING_RATE):
        self._weights = dict(initial_weights or DEFAULT_WEIGHTS)
        self._lr = lr
        self._feedback_count = 0
        self._history: List[FeedbackSample] = []
        self._lock = threading.RLock()

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def record_feedback(self, sample: FeedbackSample) -> Dict[str, float]:
        with self._lock:
            self._history.append(sample)
            self._feedback_count += 1
            
            y_true = float(sample.label)
            # Compute predicted linear sum
            pred = sum(self._weights.get(k, 0.0) * float(sample.components.get(k, 0.0)) for k in self._weights)
            # Clip prediction to avoid math issues
            pred = max(1e-4, min(1.0 - 1e-4, pred))
            
            # Gradient of MSE loss: dL/dw_k = 2 * (pred - y_true) * c_k
            error = pred - y_true
            
            for k in self._weights:
                c_k = float(sample.components.get(k, 0.0))
                grad = 2.0 * error * c_k
                self._weights[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, self._weights[k] - self._lr * grad))
                
            # Normalize onto simplex
            total = sum(self._weights.values())
            if total > 0:
                self._weights = {k: round(v / total, 4) for k, v in self._weights.items()}
                
            return dict(self._weights)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "feedback_count": self._feedback_count,
                "current_weights": self._weights,
                "learning_rate": self._lr,
            }


# Singleton instance
_learner_instance: Optional[AdaptiveWeightLearner] = None
_learner_lock = threading.Lock()


def get_adaptive_weight_learner() -> AdaptiveWeightLearner:
    global _learner_instance
    with _learner_lock:
        if _learner_instance is None:
            _learner_instance = AdaptiveWeightLearner()
    return _learner_instance
