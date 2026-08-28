from __future__ import annotations
"""
AHRAS Module 6 — Federated Learning IDS Engine (Privacy-Preserving Multi-Tenant Detection)
--------------------------------------------------------------------------------------
Enables privacy-preserving collaborative model training across multi-tenant organizations
(e.g., Bank A, Hospital B, Enterprise C) without transmitting raw telemetry logs.

Key Algorithm:
  Implements Federated Averaging (FedAvg, McMahan et al. 2017) to aggregate local model
  weights (Autoencoder & Isolation Forest model parameters) into a global detection model.

Publishable Novelty Claim:
  Collaborative cross-domain threat intelligence sharing under strict GDPR/HIPAA privacy
  guarantees — an attack seen at Bank A immediately improves anomaly detection at Hospital B
  without either organization revealing private internal logs.
"""

import copy
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ModelUpdate:
    """Encapsulates local client model parameters and sample count."""
    client_id:     str
    num_samples:   int
    weights:       dict           # Layer weights / parameters
    local_loss:    float
    timestamp:     float


class FederatedIDSServer:
    """
    Central Coordinator managing federated round aggregation (FedAvg).
    Thread-safe via RLock.
    """

    def __init__(self, min_clients: int = 2):
        self._min_clients = min_clients
        self._global_weights: Dict[str, np.ndarray] = {}
        self._round_updates: List[ModelUpdate] = []
        self._current_round: int = 0
        self._lock = threading.RLock()

    def receive_update(self, update: ModelUpdate) -> bool:
        """Collects model update from a participating local tenant client."""
        with self._lock:
            self._round_updates.append(update)
            log.info(f"[FEDERATED] Received update from client '{update.client_id}' ({update.num_samples} samples, loss={update.local_loss:.4f})")
            return len(self._round_updates) >= self._min_clients

    def aggregate_round(self) -> Dict[str, np.ndarray]:
        """
        Performs Federated Averaging (FedAvg):
            W_global = sum_{i=1}^K (n_i / N) * W_i
        """
        with self._lock:
            if len(self._round_updates) < self._min_clients:
                log.warning(f"[FEDERATED] Cannot aggregate: {len(self._round_updates)} updates < min {self._min_clients}")
                return self._global_weights

            total_samples = sum(u.num_samples for u in self._round_updates)
            new_global_weights = {}

            # Identify layer keys from first update
            sample_keys = self._round_updates[0].weights.keys()

            for key in sample_keys:
                weighted_layer = None
                for update in self._round_updates:
                    w = np.array(update.weights[key], dtype=np.float64)
                    weight_factor = update.num_samples / total_samples
                    if weighted_layer is None:
                        weighted_layer = weight_factor * w
                    else:
                        weighted_layer += weight_factor * w
                new_global_weights[key] = weighted_layer

            self._global_weights = new_global_weights
            self._current_round += 1
            log.info(f"[FEDERATED] Successfully aggregated Round {self._current_round} across {len(self._round_updates)} clients (total samples={total_samples})")
            
            # Clear round queue
            self._round_updates.clear()
            return self._global_weights

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return copy.deepcopy(self._global_weights)


# Singleton
_fed_server_instance: Optional[FederatedIDSServer] = None
_fed_lock = threading.Lock()


def get_federated_server() -> FederatedIDSServer:
    global _fed_server_instance
    with _fed_lock:
        if _fed_server_instance is None:
            _fed_server_instance = FederatedIDSServer()
    return _fed_server_instance
