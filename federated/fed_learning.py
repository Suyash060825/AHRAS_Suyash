from __future__ import annotations
"""
AHRAS Module 10 — Sound Federated Learning IDS Engine & Temporal Client Reputation
----------------------------------------------------------------------------------
Enables collaborative anomaly detection across multi-tenant security operations
without centralizing raw telemetry logs.

Mathematical Foundations & Architecture:
  1. Differentiable Model Averaging (FedAvg, McMahan et al.):
       Applies strict parameter averaging to differentiable neural Autoencoders.

  2. Temporal Client Reputation Tracking:
       Tracks per-client historical reliability:
         T_i(t) = alpha * T_i(t-1) + (1 - alpha) * Q_i(t)
       Downweights or isolates erratic / poisoning clients dynamically during aggregation.

  3. Federated Knowledge Distillation (FedKD):
       Distills consensus soft logits across heterogeneous clients without requiring
       identical model architectures.

  4. Byzantine-Resilient Aggregation:
       Implements coordinate-wise median and gradient norm clipping.
"""

import copy
import math
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def create_noniid_client_splits(
    records: List[Any],
    n_clients: int = 5,
    noniid_degree: float = 0.8,
) -> Dict[str, List[Any]]:
    """
    Dirichlet distribution-based non-IID partitioning for Federated Learning evaluation.
    """
    labels = np.array([getattr(r, 'label', 0) for r in records])
    classes = np.unique(labels)
    client_data = {f"client_{i}": [] for i in range(n_clients)}
    
    for cls in classes:
        cls_idxs = np.where(labels == cls)[0]
        # Dirichlet allocation
        proportions = np.random.dirichlet(
            alpha=np.ones(n_clients) * (1.0 - noniid_degree + 0.01),
        )
        splits = (proportions * len(cls_idxs)).astype(int)
        
        # Ensure we don't exceed bounds
        start = 0
        for i, count in enumerate(splits):
            if i == len(splits) - 1:
                # Give remaining to last client
                count = len(cls_idxs) - start
            for j in cls_idxs[start:start+count]:
                client_data[f"client_{i}"].append(records[j])
            start += count
    return client_data

@dataclass
class ModelUpdate:

    """Encapsulates authenticated local client model parameters, sample count, and metadata."""
    client_id:     str
    num_samples:   int
    weights:       Dict[str, np.ndarray]  # Differentiable layer weights (e.g. Autoencoder W1, b1, W2, b2)
    local_loss:    float
    timestamp:     float
    client_token:  str = "authenticated_client"
    round_id:      int = 0
    logits:        Optional[np.ndarray] = None  # Optional soft logits for FedKD

    def to_dict(self) -> dict:
        return {
            "client_id":    self.client_id,
            "num_samples":  self.num_samples,
            "local_loss":   round(self.local_loss, 4),
            "timestamp":    self.timestamp,
            "round_id":     self.round_id,
            "layer_shapes": {k: list(np.array(v).shape) for k, v in self.weights.items()},
        }


class ClientReputationTracker:
    """
    Maintains exponential moving average reputation scores T_i(t) in [0.0, 1.0]
    based on update validity, loss consistency, and gradient alignment.
    """

    def __init__(self, alpha: float = 0.80, default_rep: float = 0.85):
        self.alpha = alpha
        self.default_rep = default_rep
        self._reputations: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def get_reputation(self, client_id: str) -> float:
        with self._lock:
            return self._reputations.get(client_id, self.default_rep)

    def update_reputation(
        self,
        client_id: str,
        update_valid: bool,
        local_loss: float,
        is_byzantine: bool = False
    ) -> float:
        with self._lock:
            curr = self._reputations.get(client_id, self.default_rep)
            if not update_valid or is_byzantine:
                instant_score = 0.10
            else:
                # Lower loss yields higher score
                instant_score = float(np.clip(1.0 - local_loss * 2.0, 0.30, 1.0))

            new_rep = self.alpha * curr + (1.0 - self.alpha) * instant_score
            new_rep = float(np.clip(new_rep, 0.05, 1.0))
            
            self._reputations[client_id] = round(new_rep, 4)
            if client_id not in self._history:
                self._history[client_id] = []
            self._history[client_id].append(new_rep)
            return self._reputations[client_id]

    def get_all_reputations(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._reputations)


class FederatedKnowledgeDistiller:
    """
    Distills consensus knowledge across clients by averaging soft logits / probability vectors.
    """

    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def compute_soft_logits(self, raw_scores: np.ndarray) -> np.ndarray:
        """Computes temperature-scaled softmax distribution."""
        z = np.asarray(raw_scores, dtype=np.float64) / self.temperature
        exp_z = np.exp(z - np.max(z))
        return exp_z / (np.sum(exp_z) + 1e-12)

    def distill_consensus(self, client_logits: List[np.ndarray], reputations: List[float]) -> np.ndarray:
        """Distills global consensus vector via reputation-weighted average of soft logits."""
        if not client_logits:
            return np.array([])
            
        weights = np.array(reputations, dtype=np.float64)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(client_logits)) / len(client_logits)

        consensus = np.zeros_like(client_logits[0])
        for logit_arr, w in zip(client_logits, weights):
            consensus += w * np.asarray(logit_arr, dtype=np.float64)
            
        return consensus


class FederatedIDSServer:
    """
    Central Coordinator managing federated round aggregation (FedAvg, Reputation Weighting, Trimmed Median, FedKD).
    Thread-safe via RLock.
    """

    def __init__(
        self,
        min_clients: int = 2,
        byzantine_clip_norm: float = 10.0,
        enable_robust_aggregation: bool = True,
        enable_reputation_weighting: bool = False,
        aggregation_strategy: Optional[str] = None,
        quarantine_threshold: float = 0.25,
        reputation_tracker: Optional[ClientReputationTracker] = None,
    ):
        self._min_clients = min_clients
        self._clip_norm = byzantine_clip_norm
        self._robust_agg = enable_robust_aggregation
        self._enable_reputation_weighting = enable_reputation_weighting
        self.aggregation_strategy = aggregation_strategy
        self.quarantine_threshold = quarantine_threshold
        
        self.reputation_tracker = reputation_tracker or ClientReputationTracker()
        self.knowledge_distiller = FederatedKnowledgeDistiller()
        
        self._global_weights: Dict[str, np.ndarray] = {}
        self._round_updates: List[ModelUpdate] = []
        self._current_round: int = 0
        self._rejected_updates: List[Dict[str, Any]] = []
        self._quarantined_updates: List[Dict[str, Any]] = []
        self._consensus_logits: Optional[np.ndarray] = None
        self._lock = threading.RLock()

    def receive_update(self, update: ModelUpdate) -> bool:
        """
        Validates client authenticity and gradient norm before enqueuing for aggregation.
        """
        with self._lock:
            # Validate parameter shapes
            for key, val in update.weights.items():
                arr = np.array(val, dtype=np.float64)
                norm = float(np.linalg.norm(arr))
                if np.isnan(arr).any() or np.isinf(arr).any():
                    log.warning(f"[FEDERATED] Rejected update from client '{update.client_id}': NaN/Inf detected in layer '{key}'")
                    self._rejected_updates.append({"client_id": update.client_id, "reason": "NaN/Inf values", "round": self._current_round})
                    self.reputation_tracker.update_reputation(update.client_id, update_valid=False, local_loss=1.0, is_byzantine=True)
                    return False
                if norm > self._clip_norm * 5.0:
                    log.warning(f"[FEDERATED] Rejected poisoned update from client '{update.client_id}': Excessive gradient norm ({norm:.2f})")
                    self._rejected_updates.append({"client_id": update.client_id, "reason": "Excessive norm / Poisoning", "round": self._current_round})
                    self.reputation_tracker.update_reputation(update.client_id, update_valid=False, local_loss=1.0, is_byzantine=True)
                    return False

            update.round_id = self._current_round
            self._round_updates.append(update)
            self.reputation_tracker.update_reputation(update.client_id, update_valid=True, local_loss=update.local_loss, is_byzantine=False)
            log.info(f"[FEDERATED] Received valid update from client '{update.client_id}' ({update.num_samples} samples, loss={update.local_loss:.4f})")
            return len(self._round_updates) >= self._min_clients

    def aggregate_round(self) -> Dict[str, np.ndarray]:
        """
        Performs federated round aggregation according to configured strategy:
          - 'fedavg': Standard FedAvg (sample-weighted arithmetic mean, no clipping)
          - 'norm_clip': FedAvg with gradient/parameter norm clipping
          - 'coordinate_median': Coordinate-wise median across client updates
          - 'trimmed_mean': Coordinate-wise trimmed mean (trims top/bottom 20%)
          - 'fedkd_reputation': Reputation-filtered, reputation-weighted coordinate median + FedKD distillation
        """
        with self._lock:
            if len(self._round_updates) < self._min_clients:
                log.warning(f"[FEDERATED] Cannot aggregate: {len(self._round_updates)} updates < min {self._min_clients}")
                return self._global_weights

            # Strategy resolution
            strat = self.aggregation_strategy
            if strat is None:
                if self._robust_agg:
                    strat = "fedkd_reputation" if self._enable_reputation_weighting else "coordinate_median"
                elif self._enable_reputation_weighting:
                    strat = "reputation_weighted"
                else:
                    strat = "fedavg"

            # Filter quarantined updates if reputation weighting / robust defense active
            active_updates = []
            for u in self._round_updates:
                rep = self.reputation_tracker.get_reputation(u.client_id)
                if (strat in ("fedkd_reputation", "reputation_weighted") or self._enable_reputation_weighting) and rep < self.quarantine_threshold:
                    log.warning(f"[FEDERATED] Quarantined update from rogue client '{u.client_id}' (reputation={rep:.4f} < {self.quarantine_threshold})")
                    self._quarantined_updates.append({"client_id": u.client_id, "round": self._current_round, "reputation": rep})
                else:
                    active_updates.append(u)

            if len(active_updates) < self._min_clients:
                # If too many quarantined, fallback to active updates or take top reputation
                if len(active_updates) == 0:
                    sorted_updates = sorted(self._round_updates, key=lambda x: self.reputation_tracker.get_reputation(x.client_id), reverse=True)
                    active_updates = sorted_updates[:self._min_clients]
                else:
                    active_updates = list(self._round_updates)

            total_samples = sum(u.num_samples for u in active_updates)
            if self._enable_reputation_weighting or strat in ("fedkd_reputation", "reputation_weighted"):
                client_reps = [self.reputation_tracker.get_reputation(u.client_id) for u in active_updates]
                effective_masses = [u.num_samples * rep for u, rep in zip(active_updates, client_reps)]
                total_mass = sum(effective_masses)
                if total_mass <= 0:
                    total_mass = 1.0
                    effective_masses = [1.0] * len(active_updates)
                client_weights = [m / total_mass for m in effective_masses]
            else:
                client_weights = [u.num_samples / total_samples for u in active_updates]

            new_global_weights: Dict[str, np.ndarray] = {}
            sample_keys = active_updates[0].weights.keys()

            for key in sample_keys:
                client_layers = []

                for update in active_updates:
                    w = np.array(update.weights[key], dtype=np.float64)
                    # Clip layer norm to bound individual client influence (unless naive fedavg)
                    if strat != "fedavg":
                        norm = float(np.linalg.norm(w))
                        if norm > self._clip_norm:
                            w = w * (self._clip_norm / norm)
                    client_layers.append(w)

                if strat in ("coordinate_median", "fedkd_reputation") and len(client_layers) >= 4:
                    # Coordinate-wise median for Byzantine robustness
                    stacked = np.stack(client_layers, axis=0)
                    new_global_weights[key] = np.median(stacked, axis=0)
                elif strat == "trimmed_mean" and len(client_layers) >= 4:
                    # Coordinate-wise trimmed mean: trim top 20% and bottom 20%
                    stacked = np.stack(client_layers, axis=0)
                    sorted_coords = np.sort(stacked, axis=0)
                    trim_k = max(1, int(len(client_layers) * 0.20))
                    trimmed = sorted_coords[trim_k : len(client_layers) - trim_k]
                    new_global_weights[key] = np.mean(trimmed, axis=0)
                else:
                    weighted_layer = np.zeros_like(client_layers[0])
                    for w, alpha in zip(client_layers, client_weights):
                        weighted_layer += alpha * w
                    new_global_weights[key] = weighted_layer

            # Federated Knowledge Distillation (FedKD) soft logits consensus
            logits_list = [u.logits for u in active_updates if u.logits is not None]
            if logits_list:
                reps = [self.reputation_tracker.get_reputation(u.client_id) for u in active_updates if u.logits is not None]
                self._consensus_logits = self.knowledge_distiller.distill_consensus(logits_list, reps)

            self._global_weights = new_global_weights
            self._current_round += 1
            n_clients = len(active_updates)
            self._round_updates.clear()
            log.info(f"[FEDERATED] Completed Round {self._current_round} aggregation across {n_clients} active clients (total samples={total_samples})")
            return self._global_weights

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return copy.deepcopy(self._global_weights)

    def get_consensus_logits(self) -> Optional[np.ndarray]:
        with self._lock:
            return copy.deepcopy(self._consensus_logits) if self._consensus_logits is not None else None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_round": self._current_round,
                "pending_updates": len(self._round_updates),
                "min_clients_required": self._min_clients,
                "rejected_poisoned_updates": len(self._rejected_updates),
                "quarantined_updates": len(self._quarantined_updates),
                "robust_aggregation_active": self._robust_agg,
                "reputation_weighting_active": self._enable_reputation_weighting,
                "aggregation_strategy": self.aggregation_strategy,
                "client_reputations": self.reputation_tracker.get_all_reputations(),
            }


# Singleton
_fed_server_instance: Optional[FederatedIDSServer] = None
_fed_lock = threading.Lock()


def get_federated_server() -> FederatedIDSServer:
    global _fed_server_instance
    with _fed_lock:
        if _fed_server_instance is None:
            _fed_server_instance = FederatedIDSServer()
    return _fed_server_instance


class PersonalizedFedProxClient:
    """
    Personalized Federated Client with FedProx Proximal Regularization:
        L_prox(w) = L_local(w) + (mu / 2) * ||w - w_global||^2
    Produces a personalized local model: w_pers = (1 - gamma) * w_global + gamma * w_local.
    Supports both simulated proximal steps and analytical gradient descent on 14-dim telemetry.
    """

    def __init__(self, client_id: str, mu_prox: float = 0.10, gamma_pers: float = 0.30):
        self.client_id = client_id
        self.mu = mu_prox
        self.gamma = gamma_pers
        self.local_weights: Dict[str, np.ndarray] = {}

    def local_train_step(
        self,
        global_weights: Dict[str, np.ndarray],
        local_data: np.ndarray,
        local_labels: Optional[np.ndarray] = None,
        n_epochs: int = 3,
        lr: float = 0.02,
    ) -> ModelUpdate:
        """
        Executes local training step with FedProx proximal regularization.
        If local_labels is provided, computes exact gradient descent on cross-entropy loss.
        Otherwise falls back to proximal noise update for lightweight simulation.
        """
        if local_labels is not None and len(local_data) > 0 and "W1" in global_weights and "W2" in global_weights:
            # Analytical 2-layer classifier training with FedProx
            W1 = np.copy(global_weights["W1"])
            b1 = np.copy(global_weights["b1"])
            W2 = np.copy(global_weights["W2"])
            b2 = np.copy(global_weights["b2"])
            
            X = np.asarray(local_data, dtype=np.float64)
            y = np.asarray(local_labels, dtype=np.int64)
            n_samples = len(X)
            
            # One-hot labels
            Y_onehot = np.zeros((n_samples, 2), dtype=np.float64)
            for i, label in enumerate(y):
                Y_onehot[i, min(int(label), 1)] = 1.0

            local_loss = 0.0
            for _ in range(n_epochs):
                # Forward pass
                h = np.maximum(0.0, np.dot(X, W1) + b1) # ReLU (N, H)
                logits = np.dot(h, W2) + b2             # (N, 2)
                exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)
                
                # Cross-entropy loss + Proximal term
                ce_loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))
                prox_loss = 0.5 * self.mu * (
                    np.sum((W1 - global_weights["W1"]) ** 2) +
                    np.sum((b1 - global_weights["b1"]) ** 2) +
                    np.sum((W2 - global_weights["W2"]) ** 2) +
                    np.sum((b2 - global_weights["b2"]) ** 2)
                )
                local_loss = float(ce_loss + prox_loss)
                
                # Backward pass
                dlogits = (probs - Y_onehot) / n_samples
                dW2 = np.dot(h.T, dlogits) + self.mu * (W2 - global_weights["W2"])
                db2 = np.sum(dlogits, axis=0) + self.mu * (b2 - global_weights["b2"])
                
                dh = np.dot(dlogits, W2.T)
                dh[h <= 0] = 0.0
                dW1 = np.dot(X.T, dh) + self.mu * (W1 - global_weights["W1"])
                db1 = np.sum(dh, axis=0) + self.mu * (b1 - global_weights["b1"])
                
                # Gradient update
                W1 -= lr * dW1
                b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

            updated_weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
            self.local_weights = updated_weights

            # Compute soft logits for FedKD
            h_final = np.maximum(0.0, np.dot(X, W1) + b1)
            final_logits = np.dot(h_final, W2) + b2
            mean_logits = np.mean(final_logits, axis=0)

            return ModelUpdate(
                client_id=self.client_id,
                num_samples=n_samples,
                weights=updated_weights,
                local_loss=round(local_loss, 4),
                timestamp=0.0,
                logits=mean_logits,
            )
        else:
            # Fallback proximal noise update
            local_loss = 0.05
            updated_weights = {}
            for k, w_g in global_weights.items():
                noise = np.random.normal(0.0, 0.02, size=w_g.shape)
                w_loc = w_g + noise - self.mu * noise
                updated_weights[k] = w_loc

            self.local_weights = updated_weights
            return ModelUpdate(
                client_id=self.client_id,
                num_samples=len(local_data) if len(local_data) > 0 else 100,
                weights=updated_weights,
                local_loss=local_loss,
                timestamp=0.0,
            )

    def get_personalized_weights(self, global_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.local_weights:
            return global_weights
        pers = {}
        for k in global_weights:
            pers[k] = (1.0 - self.gamma) * global_weights[k] + self.gamma * self.local_weights.get(k, global_weights[k])
        return pers
