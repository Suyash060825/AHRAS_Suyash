"""
AHRAS Explanation Stability Auditor
-----------------------------------
Audits feature attribution stability under bounded input noise and perturbations,
computing top-k Jaccard rank overlap, rank correlation (Kendall tau, Spearman rho),
and empirical XAI Lipschitz continuity bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class StabilityMetrics:
    """
    Quantifies attribution invariance under input perturbations.
    """
    top_k: int
    mean_jaccard_similarity: float
    p95_instability: float
    worst_case_instability: float
    kendall_tau: float
    spearman_rho: float
    lipschitz_continuity: float
    is_stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "mean_jaccard_similarity": round(self.mean_jaccard_similarity, 4),
            "p95_instability": round(self.p95_instability, 4),
            "worst_case_instability": round(self.worst_case_instability, 4),
            "kendall_tau": round(self.kendall_tau, 4),
            "spearman_rho": round(self.spearman_rho, 4),
            "lipschitz_continuity": round(self.lipschitz_continuity, 4),
            "is_stable": self.is_stable,
        }


class ExplanationStabilityAuditor:
    """
    Audits explanation stability under stochastic and adversarial feature perturbations.
    """
    def __init__(
        self,
        noise_std: float = 0.05,
        n_perturbations: int = 50,
        min_jaccard_threshold: float = 0.80,
    ) -> None:
        self.noise_std = noise_std
        self.n_perturbations = n_perturbations
        self.min_jaccard_threshold = min_jaccard_threshold

    def evaluate_attribution_stability(
        self,
        base_features: np.ndarray,
        attribution_func: Callable[[np.ndarray], np.ndarray],
        top_k: int = 5,
        seed: int = 42,
    ) -> StabilityMetrics:
        """
        Applies zero-mean Gaussian jitter to input features and evaluates the stability
        of top-k attribution rankings and Lipschitz continuity.
        """
        rng = np.random.default_rng(seed)
        base_attr = attribution_func(base_features)
        base_topk = set(np.argsort(np.abs(base_attr))[-top_k:])

        jaccards: List[float] = []
        lipschitz_ratios: List[float] = []
        perturbed_attrs: List[np.ndarray] = []

        dim = len(base_features)
        for _ in range(self.n_perturbations):
            noise = rng.normal(0.0, self.noise_std, size=dim)
            x_pert = base_features + noise
            attr_pert = attribution_func(x_pert)
            perturbed_attrs.append(attr_pert)

            # Jaccard of top-k
            pert_topk = set(np.argsort(np.abs(attr_pert))[-top_k:])
            jacc = len(base_topk.intersection(pert_topk)) / max(1, len(base_topk.union(pert_topk)))
            jaccards.append(jacc)

            # Empirical Lipschitz: ||phi(x) - phi(x+d)||_2 / ||d||_2
            d_norm = np.linalg.norm(noise)
            attr_diff_norm = np.linalg.norm(attr_pert - base_attr)
            if d_norm > 1e-9:
                lipschitz_ratios.append(attr_diff_norm / d_norm)

        arr_jacc = np.array(jaccards)
        arr_instability = 1.0 - arr_jacc
        mean_jacc = float(np.mean(arr_jacc))
        p95_instability = float(np.percentile(arr_instability, 95))
        worst_instability = float(np.max(arr_instability))

        # Average Spearman & Kendall over perturbations
        spearman_scores = []
        kendall_scores = []
        base_ranks = np.argsort(np.argsort(-np.abs(base_attr)))

        for attr_p in perturbed_attrs:
            pert_ranks = np.argsort(np.argsort(-np.abs(attr_p)))
            # Spearman rank correlation
            d_sq = np.sum((base_ranks - pert_ranks) ** 2)
            n = len(base_ranks)
            spearman = 1.0 - (6.0 * d_sq) / max(1.0, (n * (n**2 - 1)))
            spearman_scores.append(spearman)

            # Kendall tau
            concordant, discordant = 0, 0
            for i in range(n):
                for j in range(i + 1, n):
                    sign_base = np.sign(base_ranks[i] - base_ranks[j])
                    sign_pert = np.sign(pert_ranks[i] - pert_ranks[j])
                    if sign_base * sign_pert > 0:
                        concordant += 1
                    elif sign_base * sign_pert < 0:
                        discordant += 1
            pairs = (n * (n - 1)) / 2.0
            kendall = (concordant - discordant) / max(1.0, pairs)
            kendall_scores.append(kendall)

        mean_spearman = float(np.mean(spearman_scores))
        mean_kendall = float(np.mean(kendall_scores))
        mean_lipschitz = float(np.mean(lipschitz_ratios)) if lipschitz_ratios else 0.0

        is_stable = mean_jacc >= self.min_jaccard_threshold and mean_spearman >= 0.70

        return StabilityMetrics(
            top_k=top_k,
            mean_jaccard_similarity=mean_jacc,
            p95_instability=p95_instability,
            worst_case_instability=worst_instability,
            kendall_tau=mean_kendall,
            spearman_rho=mean_spearman,
            lipschitz_continuity=mean_lipschitz,
            is_stable=is_stable,
        )
