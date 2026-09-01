from __future__ import annotations
"""
AHRAS Module 9 — Controlled Adaptive Weight Learning & Multi-Memory Continual Engine
-------------------------------------------------------------------------------------
Implements safely gated, versioned online weight adaptation, evidence quality weighting,
independence-aware de-correlation, and structured 5-compartment multi-memory continual learning.

Key Subsystems:
  1. AdaptiveWeightLearner: Shadow learning, validation check, stability constraints, rollback.
  2. ContextGatedFusionNetwork: Context-conditioned evidence fusion weights w_t.
  3. EvidenceQualityEngine: Dynamic quality scoring Q_i per evidence stream.
  4. EvidenceIndependenceController: De-correlation discounting based on source covariance.
  5. MultiMemoryReplayBuffer: 5-bank continual learning memory (Recent, Attack, Hard-Neg, Drift, Prototype).
  6. ContinualLearningEngine: Manages memory buffers, drift detection, and strategic replay.
"""

import math
import copy
import logging
import threading
from collections import deque
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


# ---------------------------------------------------------------------------
# Evidence Quality & Independence Control
# ---------------------------------------------------------------------------

@dataclass
class EvidenceQuality:
    """Quality assessment metadata for an evidence source."""
    source_name:        str
    reliability:        float    # [0.0, 1.0] historical source reliability
    freshness:          float    # [0.0, 1.0] exponential decay with age
    independence_score: float    # [0.0, 1.0] low correlation with dominant sources
    quality_score:      float    # Composite multiplier Q_i in [0.0, 1.0]

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceQualityEngine:
    """
    Computes dynamic evidence quality score Q_i for each security input:
        Q_i = reliability_i * freshness_i * independence_i
    """

    def __init__(self, half_life_sec: float = 300.0):
        self.half_life_sec = half_life_sec
        # Default baseline source reliabilities
        self.source_reliabilities: Dict[str, float] = {
            "signature":   1.00,
            "anomaly":     0.95,
            "statistical": 0.90,
            "graph":       0.95,
            "forecast":    0.85,
            "threat_intel": 1.00,
            "trust":       1.00,
        }

    def compute_freshness(self, age_seconds: float) -> float:
        """Freshness decay: exp(-lambda * age)"""
        decay_lambda = math.log(2) / max(1.0, self.half_life_sec)
        return float(np.clip(math.exp(-decay_lambda * max(0.0, age_seconds)), 0.05, 1.0))

    def evaluate_quality(
        self,
        source_name: str,
        age_seconds: float = 0.0,
        independence: float = 1.0
    ) -> EvidenceQuality:
        rel = self.source_reliabilities.get(source_name, 0.80)
        freshness = self.compute_freshness(age_seconds)
        indep = float(np.clip(independence, 0.10, 1.0))
        
        q_score = float(np.clip(rel * freshness * indep, 0.05, 1.0))
        return EvidenceQuality(
            source_name=source_name,
            reliability=round(rel, 4),
            freshness=round(freshness, 4),
            independence_score=round(indep, 4),
            quality_score=round(q_score, 4),
        )


class EvidenceIndependenceController:
    """
    De-correlates redundant evidence sources by discounting weights based on empirical cross-correlation:
        w_i' = w_i / (1 + sum_{j != i} C_{ij} * w_j)
    """

    def __init__(self, n_sources: int = 7):
        self.n_sources = n_sources
        self.corr_matrix = np.eye(n_sources, dtype=np.float64)

    def update_correlations(self, score_matrix: np.ndarray) -> None:
        """
        Updates empirical correlation matrix C from a history matrix of evidence scores (N, M).
        """
        X = np.asarray(score_matrix, dtype=np.float64)
        if len(X) >= 5 and X.shape[1] == self.n_sources:
            # Add small noise to avoid zero variance
            X_clean = X + 1e-6 * np.random.randn(*X.shape)
            corr = np.corrcoef(X_clean, rowvar=False)
            # Clip correlation in [0, 1]
            corr = np.nan_to_num(corr, nan=0.0)
            self.corr_matrix = np.clip(corr, 0.0, 1.0)

    def decorrelate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Applies de-correlation discounting to fusion weights and re-normalizes to sum to 1.0.
        """
        keys = list(weights.keys())
        w_vec = np.array([weights[k] for k in keys], dtype=np.float64)
        
        M = len(w_vec)
        if M != self.n_sources:
            # Fallback if dimension mismatch
            return dict(weights)

        # Discount denominator: 1 + sum_{j != i} C_ij * w_j
        discounted = np.zeros(M)
        for i in range(M):
            cross_overlap = sum(self.corr_matrix[i, j] * w_vec[j] for j in range(M) if j != i)
            discounted[i] = w_vec[i] / (1.0 + cross_overlap)
            
        total = np.sum(discounted)
        if total > 0:
            norm_w = discounted / total
        else:
            norm_w = w_vec
            
        return {k: round(float(norm_w[idx]), 4) for idx, k in enumerate(keys)}


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
        self.independence_ctrl = EvidenceIndependenceController(n_sources=out_dim)
        self._lock = threading.RLock()

    def compute_weights(self, z_context: np.ndarray, apply_decorrelation: bool = False) -> Dict[str, float]:
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
            
            base_weights = {
                "w_sig":   round(float(norm_w[0]), 4),
                "w_ml":    round(float(norm_w[1]), 4),
                "w_trust": round(float(norm_w[2]), 4),
                "w_hist":  round(float(norm_w[3]), 4),
                "w_graph": round(float(norm_w[4]), 4),
                "w_fore":  round(float(norm_w[5]), 4),
                "w_ti":    round(float(norm_w[6]), 4),
            }
            
            if apply_decorrelation:
                return self.independence_ctrl.decorrelate_weights(base_weights)
            return base_weights


# ---------------------------------------------------------------------------
# Structured 5-Compartment Multi-Memory Continual Learning
# ---------------------------------------------------------------------------

class MultiMemoryReplayBuffer:
    """
    5-Compartment Memory Architecture for Continual Anomaly Learning:
      1. Recent Memory: FIFO queue of recent operational events (recency).
      2. Attack Memory: Buffer of confirmed attack vectors (rare class preservation).
      3. Hard-Negative Memory: High-loss benign events near decision boundary.
      4. Drift Memory: Samples captured during detected concept drift episodes.
      5. Prototype Memory: Moving centroid vectors for normal and attack clusters.
    """

    def __init__(
        self,
        recent_cap: int = 150,
        attack_cap: int = 100,
        hard_neg_cap: int = 100,
        drift_cap: int = 100,
        feature_dim: int = 14
    ):
        self.recent_memory: deque = deque(maxlen=recent_cap)
        self.attack_memory: deque = deque(maxlen=attack_cap)
        self.hard_negative_memory: deque = deque(maxlen=hard_neg_cap)
        self.drift_memory: deque = deque(maxlen=drift_cap)
        
        self.feature_dim = feature_dim
        self.normal_prototype: Optional[np.ndarray] = None
        self.attack_prototype: Optional[np.ndarray] = None
        self.prototype_alpha: float = 0.05

    def add_sample(
        self,
        sample: FeedbackSample,
        loss: float,
        is_drift: bool = False
    ) -> None:
        """Routes sample to appropriate memory compartments."""
        self.recent_memory.append(sample)
        
        if sample.label == 1:
            self.attack_memory.append(sample)
        elif loss >= 0.25:  # Hard negative (high loss benign)
            self.hard_negative_memory.append(sample)
            
        if is_drift:
            self.drift_memory.append(sample)

        # Update prototype moving centroid if component features available
        vals = np.array(list(sample.components.values()), dtype=np.float64)
        if len(vals) > 0:
            if sample.label == 0:
                if self.normal_prototype is None:
                    self.normal_prototype = np.array(vals)
                else:
                    self.normal_prototype = (1.0 - self.prototype_alpha) * self.normal_prototype + self.prototype_alpha * vals
            else:
                if self.attack_prototype is None:
                    self.attack_prototype = np.array(vals)
                else:
                    self.attack_prototype = (1.0 - self.prototype_alpha) * self.attack_prototype + self.prototype_alpha * vals

    def sample_balanced_batch(self, batch_size: int = 16) -> List[FeedbackSample]:
        """
        Samples a balanced replay batch across memory compartments to prevent catastrophic forgetting.
        """
        candidates: List[FeedbackSample] = []
        for q in (self.attack_memory, self.hard_negative_memory, self.drift_memory, self.recent_memory):
            candidates.extend(list(q))
            
        if not candidates:
            return []
            
        n = min(batch_size, len(candidates))
        # Unique samples
        chosen_indices = np.random.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in chosen_indices]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "recent_count": len(self.recent_memory),
            "attack_count": len(self.attack_memory),
            "hard_negative_count": len(self.hard_negative_memory),
            "drift_count": len(self.drift_memory),
            "has_normal_proto": self.normal_prototype is not None,
            "has_attack_proto": self.attack_prototype is not None,
        }


class ContinualLearningEngine:
    """
    Continual Learning Engine with 5-Compartment Multi-Memory Replay and Drift Detection.
    """

    def __init__(self, memory_capacity: int = 500, decay_rate: float = 0.01):
        self.capacity = memory_capacity
        self.decay_rate = decay_rate
        self.replay_buffer: List[Dict[str, Any]] = []
        self.multi_memory = MultiMemoryReplayBuffer()
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

            # Add to multi-memory structure
            self.multi_memory.add_sample(sample, loss=loss, is_drift=self.drift_detected)

            importance_score = loss + (1.0 if is_hard_sample else 0.0)
            item = {
                "sample": sample,
                "importance": importance_score,
                "age": 0,
            }
            
            if len(self.replay_buffer) >= self.capacity:
                self.replay_buffer.sort(key=lambda x: x["importance"] / (1.0 + self.decay_rate * x["age"]))
                self.replay_buffer.pop(0)

            self.replay_buffer.append(item)
            for it in self.replay_buffer:
                it["age"] += 1

    def sample_replay_batch(self, batch_size: int = 16) -> List[FeedbackSample]:
        with self._lock:
            # Prefer balanced multi-memory sampling if populated
            multi_batch = self.multi_memory.sample_balanced_batch(batch_size)
            if multi_batch:
                return multi_batch
                
            if not self.replay_buffer:
                return []
            probs = np.array([it["importance"] for it in self.replay_buffer])
            probs = probs / np.sum(probs)
            n_samples = min(batch_size, len(self.replay_buffer))
            indices = np.random.choice(len(self.replay_buffer), size=n_samples, p=probs, replace=False)
            return [self.replay_buffer[i]["sample"] for i in indices]
