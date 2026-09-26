"""
AHRAS Explainer Consensus Engine
--------------------------------
Evaluates cross-method explanation consensus across heterogeneous attribution
algorithms (Causal DAG, Counterfactuals, Gradient Sensitivity, Shapley approximation).
Computes the Explainer Disagreement Index (EDI) and extracts unanimous consensus features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ConsensusReport:
    """
    Consensus analysis across heterogeneous explainers.
    """
    explainer_names: List[str]
    top_k: int
    mean_pairwise_jaccard: float
    explainer_disagreement_index: float  # 1.0 - mean_pairwise_jaccard [0, 1]
    consensus_features: List[int]        # Features agreed upon by majority
    unanimous_features: List[int]        # Features agreed upon by 100% of methods
    high_consensus: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explainer_names": self.explainer_names,
            "top_k": self.top_k,
            "mean_pairwise_jaccard": round(self.mean_pairwise_jaccard, 4),
            "explainer_disagreement_index": round(self.explainer_disagreement_index, 4),
            "consensus_features": self.consensus_features,
            "unanimous_features": self.unanimous_features,
            "high_consensus": self.high_consensus,
        }


class ExplainerConsensusEngine:
    """
    Aggregates and compares attributions from multiple explainer algorithms.
    """
    def __init__(self, majority_threshold: float = 0.60) -> None:
        self.majority_threshold = majority_threshold

    def evaluate_consensus(
        self,
        attributions_by_method: Dict[str, np.ndarray],
        top_k: int = 5,
    ) -> ConsensusReport:
        """
        Calculates pairwise Jaccard overlap, EDI, and consensus feature subsets.
        """
        method_names = list(attributions_by_method.keys())
        n_methods = len(method_names)
        if n_methods < 2:
            single_topk = list(np.argsort(-np.abs(list(attributions_by_method.values())[0]))[:top_k])
            return ConsensusReport(
                explainer_names=method_names,
                top_k=top_k,
                mean_pairwise_jaccard=1.0,
                explainer_disagreement_index=0.0,
                consensus_features=single_topk,
                unanimous_features=single_topk,
                high_consensus=True,
            )

        topk_sets = {
            m: set(np.argsort(-np.abs(attributions_by_method[m]))[:top_k])
            for m in method_names
        }

        # Compute pairwise Jaccard similarities
        jaccards = []
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                set_a = topk_sets[method_names[i]]
                set_b = topk_sets[method_names[j]]
                jacc = len(set_a.intersection(set_b)) / max(1, len(set_a.union(set_b)))
                jaccards.append(jacc)

        mean_jaccard = float(np.mean(jaccards))
        edi = max(0.0, 1.0 - mean_jaccard)

        # Count occurrences of each feature in top-k
        feature_counts: Dict[int, int] = {}
        for s in topk_sets.values():
            for feat in s:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        majority_req = int(np.ceil(self.majority_threshold * n_methods))
        consensus_feats = [feat for feat, cnt in feature_counts.items() if cnt >= majority_req]
        unanimous_feats = [feat for feat, cnt in feature_counts.items() if cnt == n_methods]

        consensus_feats.sort()
        unanimous_feats.sort()

        high_consensus = mean_jaccard >= 0.70 and len(consensus_feats) >= int(0.6 * top_k)

        return ConsensusReport(
            explainer_names=method_names,
            top_k=top_k,
            mean_pairwise_jaccard=mean_jaccard,
            explainer_disagreement_index=edi,
            consensus_features=consensus_feats,
            unanimous_features=unanimous_feats,
            high_consensus=high_consensus,
        )
