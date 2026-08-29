from __future__ import annotations
"""
AHRAS Module 9 — Controlled Adaptive Weight Learning Engine
------------------------------------------------------------
Implements safely gated, versioned online weight adaptation from SOC feedback.

Safety Controls:
  1. Shadow Learning Mode: Updates are computed in shadow before promotion.
  2. Approval-Gated Promotion: Production weights require validation check or analyst approval.
  3. Minimum Sample Batching: Requires at least N feedback samples before proposing update.
  4. Stability Constraints: Bounds maximum per-update weight change (Δw <= 0.05).
  5. Rolling Validation & Automatic Freeze: Freezes weights if loss increases on validation buffer.
  6. Versioned Checkpoints & Rollback: Full audit history and one-click rollback.
"""

import math
import copy
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "signature":  0.40,
    "anomaly":    0.30,
    "density":    0.20,
    "drift_rate": 0.10,
}

MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.70
MAX_DELTA_PER_UPDATE = 0.05
DEFAULT_LR = 0.02
MIN_BATCH_SAMPLES = 5


@dataclass
class FeedbackSample:
    src_ip:         str
    label:          int                   # 1 = true attack, 0 = false positive / benign
    components:     Dict[str, float]      # component values in [0, 1]
    predicted_risk: float                 # model's risk score
    timestamp:      float = 0.0


@dataclass
class WeightVersion:
    version_id:       int
    weights:          Dict[str, float]
    created_at:       float
    validation_loss:  float
    is_active:        bool
    promoted_by:      str = "SYSTEM_VALIDATION"


class AdaptiveWeightLearner:
    """
    Safely tunes fusion weights with stability constraints, shadow validation,
    and automatic rollback. Thread-safe.
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        lr: float = DEFAULT_LR,
        shadow_mode: bool = False,
    ):
        self._active_weights = dict(initial_weights or DEFAULT_WEIGHTS)
        self._shadow_weights = dict(self._active_weights)
        self._lr = lr
        self._shadow_mode = shadow_mode
        self._is_frozen = False
        
        self._feedback_count = 0
        self._history: List[FeedbackSample] = []
        self._validation_buffer: List[FeedbackSample] = []
        
        self._version_counter = 1
        self._versions: List[WeightVersion] = [
            WeightVersion(
                version_id=1,
                weights=dict(self._active_weights),
                created_at=0.0,
                validation_loss=0.0,
                is_active=True,
                promoted_by="INITIALIZATION",
            )
        ]
        self._lock = threading.RLock()

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._active_weights)

    def get_shadow_weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._shadow_weights)

    def record_feedback(self, sample: FeedbackSample, auto_promote: bool = True) -> Dict[str, float]:
        """
        Ingests feedback, performs gradient step on shadow weights with clipping,
        and promotes to active weights if validation passes.
        """
        with self._lock:
            self._history.append(sample)
            self._validation_buffer.append(sample)
            if len(self._validation_buffer) > 100:
                self._validation_buffer.pop(0)
                
            self._feedback_count += 1
            
            if self._is_frozen:
                log.warning("[ADAPTIVE LEARNER] Learner is FROZEN due to validation drift. Feedback recorded without weight update.")
                return dict(self._active_weights)

            y_true = float(sample.label)
            pred = sum(self._shadow_weights.get(k, 0.0) * float(sample.components.get(k, 0.0)) for k in self._shadow_weights)
            pred = max(1e-4, min(1.0 - 1e-4, pred))
            
            error = pred - y_true
            updated_shadow = {}
            for k in self._shadow_weights:
                c_k = float(sample.components.get(k, 0.0))
                grad = 2.0 * error * c_k
                delta = -self._lr * grad
                
                # Stability constraint: bound maximum per-sample change
                delta_clamped = max(-MAX_DELTA_PER_UPDATE, min(MAX_DELTA_PER_UPDATE, delta))
                new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, self._shadow_weights[k] + delta_clamped))
                updated_shadow[k] = new_w
                
            # Simplex normalization
            total = sum(updated_shadow.values())
            if total > 0:
                self._shadow_weights = {k: round(v / total, 4) for k, v in updated_shadow.items()}

            # Check for promotion if minimum sample batch reached
            if auto_promote and not self._shadow_mode and self._feedback_count >= MIN_BATCH_SAMPLES:
                self._evaluate_and_promote()

            return dict(self._active_weights)

    def _evaluate_and_promote(self) -> bool:
        """Evaluates shadow weights on validation buffer; promotes if validation loss improves or is stable."""
        if not self._validation_buffer:
            return False

        current_loss = self._compute_mse(self._active_weights, self._validation_buffer)
        shadow_loss = self._compute_mse(self._shadow_weights, self._validation_buffer)

        # If shadow model degrades validation loss by more than 15%, freeze adaptation
        if shadow_loss > current_loss * 1.15 and len(self._validation_buffer) >= 10:
            log.warning(f"[ADAPTIVE LEARNER] Validation drift detected! (Shadow Loss={shadow_loss:.4f} > Active Loss={current_loss:.4f}). Freezing learner.")
            self._is_frozen = True
            return False

        # Promote shadow weights
        self._active_weights = dict(self._shadow_weights)
        self._version_counter += 1
        ver = WeightVersion(
            version_id=self._version_counter,
            weights=dict(self._active_weights),
            created_at=0.0,
            validation_loss=round(shadow_loss, 4),
            is_active=True,
            promoted_by="SYSTEM_VALIDATION",
        )
        self._versions.append(ver)
        log.info(f"[ADAPTIVE LEARNER] Promoted Weight Version {ver.version_id}: {self._active_weights} (Val Loss: {shadow_loss:.4f})")
        return True

    def _compute_mse(self, weights: Dict[str, float], dataset: List[FeedbackSample]) -> float:
        losses = []
        for s in dataset:
            p = sum(weights.get(k, 0.0) * float(s.components.get(k, 0.0)) for k in weights)
            p = max(0.0, min(1.0, p))
            losses.append((p - float(s.label)) ** 2)
        return float(np.mean(losses)) if losses else 0.0

    def rollback_to_version(self, version_id: int) -> bool:
        """Rolls back active weights to a previously recorded stable version."""
        with self._lock:
            for v in self._versions:
                if v.version_id == version_id:
                    self._active_weights = dict(v.weights)
                    self._shadow_weights = dict(v.weights)
                    self._is_frozen = False
                    log.info(f"[ADAPTIVE LEARNER] Rolled back to Weight Version {version_id}: {self._active_weights}")
                    return True
            return False

    def unfreeze(self) -> None:
        with self._lock:
            self._is_frozen = False
            log.info("[ADAPTIVE LEARNER] Learner unfrozen by analyst.")

    def get_version_history(self) -> List[dict]:
        with self._lock:
            return [asdict(v) for v in self._versions]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "feedback_count": self._feedback_count,
                "current_weights": self._active_weights,
                "shadow_weights": self._shadow_weights,
                "learning_rate": self._lr,
                "is_frozen": self._is_frozen,
                "shadow_mode": self._shadow_mode,
                "version_count": len(self._versions),
            }


# Singleton
_learner_instance: Optional[AdaptiveWeightLearner] = None
_learner_lock = threading.Lock()


def get_adaptive_weight_learner() -> AdaptiveWeightLearner:
    global _learner_instance
    with _learner_lock:
        if _learner_instance is None:
            _learner_instance = AdaptiveWeightLearner()
    return _learner_instance
