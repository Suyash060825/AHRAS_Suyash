from __future__ import annotations
"""
AHRAS Module — Dynamic Context-Conditioned Feature Selection
------------------------------------------------------------
Implements a learned, context-conditioned feature selector:

    m_t = Sigmoid(W_sel · z_context + b_sel)  ∈ [0, 1]^D

Features are dynamically masked based on the current entity/network context z_context,
enabling AHRAS to attend to relevant indicators during attacks (e.g., port scan vs credential brute force)
while attenuating noisy irrelevant features.
"""

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class DynamicFeatureSelector:
    """
    Learns and applies context-dependent feature importance masks.
    """

    def __init__(self, n_features: int = 14, context_dim: int = 8, seed: int = 42):
        self.n_features = n_features
        self.context_dim = context_dim
        
        rng = np.random.default_rng(seed)
        # Initialize linear gate weights
        self.W_sel = rng.normal(0.0, 0.1, size=(context_dim, n_features))
        self.b_sel = np.zeros(n_features)

    def compute_mask(self, z_context: np.ndarray) -> np.ndarray:
        """
        Computes continuous feature mask m_t ∈ [0, 1]^n_features given context vector z_context.
        """
        z = np.asarray(z_context, dtype=np.float64)
        if z.ndim == 1:
            logits = z @ self.W_sel + self.b_sel
        else:
            logits = z @ self.W_sel + self.b_sel
        # Sigmoid activation
        mask = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
        return mask

    def apply_mask(self, x: np.ndarray, z_context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies context mask to raw feature vector x:
            x_masked = x ⊙ m_t
        Returns (x_masked, mask).
        """
        mask = self.compute_mask(z_context)
        x_arr = np.asarray(x, dtype=np.float64)
        x_masked = x_arr * mask
        return x_masked, mask

    def train_step(
        self,
        X_batch: np.ndarray,
        z_batch: np.ndarray,
        target_importance: np.ndarray,
        lr: float = 0.01
    ) -> float:
        """
        Gradient descent step to align dynamic mask with observed feature importances or reconstruction utility.
        Loss = Mean Squared Error(mask, target_importance) + L1 sparsity penalty
        """
        X = np.asarray(X_batch, dtype=np.float64)
        Z = np.asarray(z_batch, dtype=np.float64)
        targets = np.asarray(target_importance, dtype=np.float64)
        
        N = len(Z)
        if N == 0:
            return 0.0
            
        logits = Z @ self.W_sel + self.b_sel
        masks = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
        
        # Error and gradient w.r.t logits: dL/dlogit = (mask - target) * mask * (1 - mask)
        diff = masks - targets
        d_logits = diff * masks * (1.0 - masks)
        
        grad_W = (Z.T @ d_logits) / N
        grad_b = np.mean(d_logits, axis=0)
        
        # Sparsity regularization gradient
        grad_W += 1e-4 * np.sign(self.W_sel)
        
        # Gradient update with clipping
        grad_W = np.clip(grad_W, -1.0, 1.0)
        grad_b = np.clip(grad_b, -1.0, 1.0)
        
        self.W_sel -= lr * grad_W
        self.b_sel -= lr * grad_b
        
        loss = float(np.mean(diff ** 2))
        return loss
