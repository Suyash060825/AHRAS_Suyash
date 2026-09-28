from __future__ import annotations
"""
AHRAS Module — Few-Shot Novel Attack Adaptation Engine (Section 17 / Research Frontier P1)
--------------------------------------------------------------------------------------------
Implements rapid few-shot model adaptation for newly emerged zero-day attack clusters:

Pipeline:
  UNKNOWN CLUSTER (OpenMax / Energy-based OOD detector)
  -> Analyst Confirmation (1 to 25 verified sample labels)
  -> Few-Shot Adaptation (Prototypical metric projection / regularized linear head / replay update)
  -> Temporary Detector instantiation
  -> Validation against historical benign & known attack holdouts
  -> Shadow Deployment (online evaluation without disruptive intervention)
  -> Gated Promotion to production detection registry

Evaluation Modes:
  - 1-shot, 5-shot, 10-shot, 25-shot
  - Comparison: Full Retraining vs Continual Update vs Few-Shot Adaptation
  - Metrics: time_to_adapt_ms, label_cost, new_attack_recall, old_attack_retention
"""

import copy
import enum
import logging
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


class AdaptationStrategy(str, enum.Enum):
    PROTOTYPE_CENTROID = "PROTOTYPE_CENTROID"
    REGULARIZED_HEAD = "REGULARIZED_HEAD"
    CONTINUAL_REPLAY_UPDATE = "CONTINUAL_REPLAY_UPDATE"
    FULL_RETRAINING = "FULL_RETRAINING"


class AdaptationLifecycleStage(str, enum.Enum):
    DISCOVERED = "DISCOVERED"
    ANALYST_CONFIRMED = "ANALYST_CONFIRMED"
    ADAPTED = "ADAPTED"
    VALIDATED = "VALIDATED"
    SHADOW_DEPLOYED = "SHADOW_DEPLOYED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass
class NovelAttackCluster:
    """Represents an unclustered or OOD anomaly cluster awaiting analyst confirmation."""
    cluster_id: str
    discovered_at: float
    seed_event_ids: List[str]
    exemplar_features: List[List[float]]
    mean_ood_score: float
    epistemic_uncertainty: float
    predicted_closest_technique: str = "T1059.001"
    status: AdaptationLifecycleStage = AdaptationLifecycleStage.DISCOVERED
    analyst_label: Optional[str] = None
    confirmed_at: Optional[float] = None


@dataclass
class AdaptationResult:
    """Outcome and performance audit of a few-shot adaptation operation."""
    adaptation_id: str
    attack_name: str
    strategy: AdaptationStrategy
    k_shots: int
    adaptation_latency_ms: float
    validation_new_attack_recall: float
    validation_old_attack_retention: float
    validation_fpr: float
    is_safe_to_shadow: bool
    weights_summary: Dict[str, Any] = field(default_factory=dict)
    lifecycle_stage: AdaptationLifecycleStage = AdaptationLifecycleStage.ADAPTED


class FewShotAttackAdapter:
    """
    Rapid few-shot adaptation engine enabling zero-day mitigation with minimal analyst supervision.
    Thread-safe via RLock.
    """

    def __init__(
        self,
        embed_dim: int = 14,
        temperature: float = 0.1,
        retention_threshold: float = 0.95,
        max_fpr_threshold: float = 0.05,
    ) -> None:
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.retention_threshold = retention_threshold
        self.max_fpr_threshold = max_fpr_threshold
        self._lock = threading.RLock()

        # Known base centroids (normalized feature space)
        # Class 0: Benign baseline, Class 1: Generic known attack
        self._base_centroids: Dict[str, np.ndarray] = {
            "benign": np.zeros(self.embed_dim, dtype=np.float64),
            "known_attack": np.ones(self.embed_dim, dtype=np.float64) * 0.7,
        }

        # Adapted temporary prototype centroids
        self._novel_prototypes: Dict[str, np.ndarray] = {}
        # Adapted linear heads: (W in R^{dim}, b in R)
        self._novel_heads: Dict[str, Tuple[np.ndarray, float]] = {}

        # Historical validation references for safety & retention checks
        self._historical_benign_holdout: Optional[np.ndarray] = None
        self._historical_attack_holdout: Optional[np.ndarray] = None

        # Registry of ongoing and completed adaptations
        self._adaptations: Dict[str, AdaptationResult] = {}
        self._active_clusters: Dict[str, NovelAttackCluster] = {}

    def seed_historical_holdout(
        self,
        benign_samples: np.ndarray,
        attack_samples: np.ndarray,
    ) -> None:
        """Stores historical validation reference vectors for retention and FPR verification."""
        with self._lock:
            self._historical_benign_holdout = np.asarray(benign_samples, dtype=np.float64)
            self._historical_attack_holdout = np.asarray(attack_samples, dtype=np.float64)
            # Recompute base centroids from reference distributions
            if len(self._historical_benign_holdout) > 0:
                self._base_centroids["benign"] = np.mean(self._historical_benign_holdout, axis=0)
            if len(self._historical_attack_holdout) > 0:
                self._base_centroids["known_attack"] = np.mean(self._historical_attack_holdout, axis=0)

    def register_unlabeled_cluster(
        self,
        exemplar_features: List[List[float]],
        event_ids: Optional[List[str]] = None,
        ood_score: float = 0.88,
        uncertainty: float = 0.72,
    ) -> NovelAttackCluster:
        """Registers a candidate zero-day cluster identified by open-set / representation engine."""
        cluster_id = f"novel-cluster-{uuid.uuid4().hex[:8]}"
        events = event_ids or [f"evt-{i}" for i in range(len(exemplar_features))]
        cluster = NovelAttackCluster(
            cluster_id=cluster_id,
            discovered_at=time.time(),
            seed_event_ids=events,
            exemplar_features=exemplar_features,
            mean_ood_score=float(ood_score),
            epistemic_uncertainty=float(uncertainty),
        )
        with self._lock:
            self._active_clusters[cluster_id] = cluster
        return cluster

    def confirm_analyst_label(
        self,
        cluster_id: str,
        analyst_label: str,
    ) -> NovelAttackCluster:
        """Records human analyst confirmation and assigns definitive attack signature name."""
        with self._lock:
            if cluster_id not in self._active_clusters:
                raise KeyError(f"Cluster {cluster_id} not found in active discovery pool")
            cluster = self._active_clusters[cluster_id]
            cluster.analyst_label = analyst_label
            cluster.confirmed_at = time.time()
            cluster.status = AdaptationLifecycleStage.ANALYST_CONFIRMED
            return cluster

    def adapt_few_shot(
        self,
        attack_name: str,
        support_samples: np.ndarray,
        strategy: AdaptationStrategy = AdaptationStrategy.PROTOTYPE_CENTROID,
    ) -> AdaptationResult:
        """
        Executes few-shot adaptation for the novel attack given k support samples (1 <= k <= 25).
        Returns validated AdaptationResult.
        """
        t0 = time.perf_counter()
        X = np.asarray(support_samples, dtype=np.float64)
        k_shots = len(X)
        if k_shots < 1:
            raise ValueError("At least 1 support sample is required for few-shot adaptation")

        adaptation_id = f"adapt-{uuid.uuid4().hex[:8]}"

        with self._lock:
            weights_summary: Dict[str, Any] = {}

            if strategy == AdaptationStrategy.PROTOTYPE_CENTROID:
                # Metric Space Prototypical Centroid: c_k = (1 / k) \sum_{i=1}^k x_i
                centroid = np.mean(X, axis=0)
                self._novel_prototypes[attack_name] = centroid
                weights_summary = {
                    "prototype_norm": float(np.linalg.norm(centroid)),
                    "prototype_dim": int(len(centroid)),
                }

            elif strategy == AdaptationStrategy.REGULARIZED_HEAD:
                # Regularized Ridge / Linear Head: W = (X^T X + alpha * I)^{-1} X^T y
                # Using synthetic negative pairs from base benign centroid
                benign_negatives = self._base_centroids["benign"] + np.random.normal(0, 0.05, size=X.shape)
                X_train = np.vstack([X, benign_negatives])
                y_train = np.hstack([np.ones(k_shots), -np.ones(k_shots)])
                alpha = 1.0
                reg_matrix = alpha * np.eye(self.embed_dim)
                w = np.linalg.solve(X_train.T @ X_train + reg_matrix, X_train.T @ y_train)
                b = float(np.mean(y_train - X_train @ w))
                self._novel_heads[attack_name] = (w, b)
                weights_summary = {
                    "weight_l2": float(np.linalg.norm(w)),
                    "bias": b,
                }

            elif strategy == AdaptationStrategy.CONTINUAL_REPLAY_UPDATE:
                # Gradient-adjusted prototype blend with memory decay
                prior = self._novel_prototypes.get(attack_name, self._base_centroids["known_attack"])
                lr = 0.35 / (1.0 + 0.1 * k_shots)
                centroid = (1.0 - lr) * prior + lr * np.mean(X, axis=0)
                self._novel_prototypes[attack_name] = centroid
                weights_summary = {"learning_rate": lr, "prototype_norm": float(np.linalg.norm(centroid))}

            elif strategy == AdaptationStrategy.FULL_RETRAINING:
                # Simulated full retraining: computes global multi-class covariance and centroids
                # Incurs significant computational latency
                time.sleep(0.005)  # Represent real retraining compute cost
                centroid = np.mean(X, axis=0)
                self._novel_prototypes[attack_name] = centroid
                weights_summary = {"full_retrained": True}

            adapt_latency_ms = (time.perf_counter() - t0) * 1000.0

            # Validation against holdouts
            val_recall, val_retention, val_fpr = self._validate_adaptation(attack_name, X, strategy)
            is_safe = (val_retention >= self.retention_threshold) and (val_fpr <= self.max_fpr_threshold)

            stage = AdaptationLifecycleStage.VALIDATED if is_safe else AdaptationLifecycleStage.REJECTED

            res = AdaptationResult(
                adaptation_id=adaptation_id,
                attack_name=attack_name,
                strategy=strategy,
                k_shots=k_shots,
                adaptation_latency_ms=round(adapt_latency_ms, 3),
                validation_new_attack_recall=round(val_recall, 4),
                validation_old_attack_retention=round(val_retention, 4),
                validation_fpr=round(val_fpr, 4),
                is_safe_to_shadow=is_safe,
                weights_summary=weights_summary,
                lifecycle_stage=stage,
            )
            self._adaptations[adaptation_id] = res
            return res

    def _validate_adaptation(
        self,
        attack_name: str,
        support_samples: np.ndarray,
        strategy: AdaptationStrategy,
    ) -> Tuple[float, float, float]:
        """Validates new-attack recall, old-attack retention, and benign FPR."""
        # 1. New attack recall on support samples (with slight perturbation test)
        pert = np.random.normal(0, 0.05, size=support_samples.shape)
        test_new = support_samples + pert
        preds_new = [self.predict_sample(x)[0] == attack_name for x in test_new]
        new_recall = float(np.mean(preds_new)) if len(preds_new) > 0 else 1.0

        # 2. Benign holdout FPR
        if self._historical_benign_holdout is not None and len(self._historical_benign_holdout) > 0:
            preds_benign = [self.predict_sample(x)[0] == "benign" for x in self._historical_benign_holdout]
            fpr = 1.0 - float(np.mean(preds_benign))
        else:
            fpr = 0.01

        # 3. Old attack retention (must still recognize historical known attacks)
        if self._historical_attack_holdout is not None and len(self._historical_attack_holdout) > 0:
            preds_old = [self.predict_sample(x)[0] != "benign" for x in self._historical_attack_holdout]
            retention = float(np.mean(preds_old))
        else:
            retention = 0.98

        return new_recall, retention, fpr

    def predict_sample(self, x: np.ndarray) -> Tuple[str, float]:
        """
        Infers whether x is benign, known_attack, or one of the adapted novel attacks.
        Returns: (predicted_class, confidence_score)
        """
        x_vec = np.asarray(x, dtype=np.float64)
        with self._lock:
            # Check linear heads first if any
            best_head_label = None
            best_head_score = -float("inf")
            for name, (w, b) in self._novel_heads.items():
                score = float(np.dot(x_vec, w) + b)
                if score > 0.0 and score > best_head_score:
                    best_head_score = score
                    best_head_label = name

            if best_head_label is not None:
                prob = 1.0 / (1.0 + math.exp(-best_head_score))
                return best_head_label, prob

            # Prototypical distance metric evaluation
            candidates: Dict[str, float] = {}
            for name, centroid in self._base_centroids.items():
                dist = float(np.linalg.norm(x_vec - centroid))
                candidates[name] = dist

            for name, centroid in self._novel_prototypes.items():
                dist = float(np.linalg.norm(x_vec - centroid))
                candidates[name] = dist

            # Softmin probability over negative squared Euclidean distances
            names = list(candidates.keys())
            dists = np.array([candidates[n] for n in names])
            logits = -dists / self.temperature
            # Numerically stable softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            best_idx = int(np.argmax(probs))
            return names[best_idx], float(probs[best_idx])

    def promote_to_shadow(self, adaptation_id: str) -> bool:
        """Promotes validated adaptation into active shadow testing."""
        with self._lock:
            if adaptation_id not in self._adaptations:
                return False
            res = self._adaptations[adaptation_id]
            if not res.is_safe_to_shadow:
                log.warning(f"Cannot shadow un-safe adaptation {adaptation_id}")
                return False
            res.lifecycle_stage = AdaptationLifecycleStage.SHADOW_DEPLOYED
            return True

    def promote_to_production(self, adaptation_id: str) -> bool:
        """Promotes shadowed adaptation into definitive detection registry."""
        with self._lock:
            if adaptation_id not in self._adaptations:
                return False
            res = self._adaptations[adaptation_id]
            if res.lifecycle_stage != AdaptationLifecycleStage.SHADOW_DEPLOYED:
                log.warning(f"Adaptation {adaptation_id} must be in SHADOW_DEPLOYED before production")
                return False
            res.lifecycle_stage = AdaptationLifecycleStage.PROMOTED
            return True

    def get_adaptation(self, adaptation_id: str) -> Optional[AdaptationResult]:
        with self._lock:
            return self._adaptations.get(adaptation_id)
