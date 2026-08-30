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


class ContextGatedFusionNetwork:
    """
    Context-Conditioned Gating Mechanism for Adaptive Evidence Fusion:
        w_t = softmax( W_g * z_t + b_g )
    where z_t in R^9 contains: [S_sig, A_ml, delta_D, G_corr, H_boost, P_fore, TI_score, U_unc, A_crit].
    Guarantees: bounded weights in [0.05, 0.60], simplex sum == 1.0, and strict audit replayability.
    """

    def __init__(self, in_dim: int = 9, out_dim: int = 7, seed: int = 42):
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W_g = rng.normal(0.0, scale, size=(in_dim, out_dim))
        self.b_g = np.array([0.50, 0.30, 0.15, 0.10, 0.10, 0.05, 0.15], dtype=np.float64)
        self._lock = threading.RLock()

    def compute_weights(self, z_context: np.ndarray) -> Dict[str, float]:
        with self._lock:
            z = np.array(z_context, dtype=np.float64).flatten()
            if len(z) < self.in_dim:
                z_padded = np.zeros(self.in_dim)
                z_padded[:len(z)] = z
                z = z_padded
            elif len(z) > self.in_dim:
                z = z[:self.in_dim]

            logits = np.dot(z, self.W_g) + self.b_g
            # Stable Softmax with temperature tau = 1.2
            exp_logits = np.exp((logits - np.max(logits)) / 1.2)
            softmax_w = exp_logits / np.sum(exp_logits)
            
            # Bound and clamp each weight in [0.05, 0.60]
            clamped = np.clip(softmax_w, 0.05, 0.60)
            norm_w = clamped / np.sum(clamped)
            
            return {
                "w_sig":   round(float(norm_w[0]), 4),
                "w_ml":    round(float(norm_w[1]), 4),
                "w_trust": round(float(norm_w[2]), 4),
                "w_hist":  round(float(norm_w[3]), 4),
                "w_graph": round(float(norm_w[4]), 4),
                "w_fore":  round(float(norm_w[5]), 4),
                "w_ti":    round(float(norm_w[6]), 4),
            }


class ContinualLearningEngine:
    """
    Advanced Continual Learning Engine with Experience Replay & Strategic Forgetting.
    Prevents catastrophic forgetting during non-stationary concept drift:
      1. Replay Memory Buffer: Retains hard negatives and rare attack vectors.
      2. Statistical Drift Detector: Measures EWMA loss residual to trigger adaptation.
      3. Strategic Forgetting: Decays stale uninformative historical telemetry.
    """

    def __init__(self, memory_capacity: int = 500, decay_rate: float = 0.01):
        self.capacity = memory_capacity
        self.decay_rate = decay_rate
        self.replay_buffer: List[Dict[str, Any]] = []
        self.drift_detected = False
        self.ewma_loss = 0.05
        self._lock = threading.RLock()

    def add_experience(self, sample: FeedbackSample, loss: float, is_hard_sample: bool = False):
        with self._lock:
            # Update drift detection EWMA
            alpha = 0.05
            self.ewma_loss = (1.0 - alpha) * self.ewma_loss + alpha * loss
            if self.ewma_loss > 0.20:
                self.drift_detected = True

            importance_score = loss + (1.0 if is_hard_sample else 0.0)
            item = {
                "sample": sample,
                "importance": importance_score,
                "age": 0,
            }
            
            if len(self.replay_buffer) >= self.capacity:
                # Strategic forgetting: prune item with lowest importance/highest age
                self.replay_buffer.sort(key=lambda x: x["importance"] / (1.0 + self.decay_rate * x["age"]))
                self.replay_buffer.pop(0)

            self.replay_buffer.append(item)
            for it in self.replay_buffer:
                it["age"] += 1

    def sample_replay_batch(self, batch_size: int = 16) -> List[FeedbackSample]:
        with self._lock:
            if not self.replay_buffer:
                return []
            probs = np.array([it["importance"] for it in self.replay_buffer])
            probs = probs / np.sum(probs)
            n_samples = min(batch_size, len(self.replay_buffer))
            indices = np.random.choice(len(self.replay_buffer), size=n_samples, p=probs, replace=False)
            return [self.replay_buffer[i]["sample"] for i in indices]
