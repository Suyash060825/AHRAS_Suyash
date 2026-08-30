r"""
AHRAS Module — Attack-Path & Multi-Hop Episode Reasoning Engine
---------------------------------------------------------------
Evaluates multi-hop attack propagation paths across the entity graph using sound probabilistic aggregation:

  1. Multi-Hop Path Risk (Noisy-OR Formulation):
       Given an attack path P = (v_1, e_1, v_2, ..., v_k):
         R_P = 1 - \prod_{i=1}^k (1 - R(v_i))

  2. Episode Embedding & Composite Risk:
       Aggregates individual entity embeddings in an attack episode via mean-pooling and an MLP projection:
         z_episode = MeanPool({ z_{v} : v \in Episode })
         R_episode = Sigmoid(W_ep · z_episode + b_ep)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AttackPath:
    """Represents an evaluated multi-hop lateral movement / attack sequence."""
    path_nodes:     List[str]
    node_risks:     List[float]
    path_risk:      float            # Probabilistic Noisy-OR composite risk in [0, 1]
    hop_count:      int
    critical_node:  str              # Node with highest individual risk on path
    
    def to_dict(self) -> dict:
        return asdict(self)


class AttackPathReasoner:
    """
    Computes graph path risk and multi-entity attack episode threat scores.
    """

    def __init__(self, embed_dim: int = 8, seed: int = 42):
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        self.W_ep = rng.normal(0.0, np.sqrt(2.0 / embed_dim), size=(embed_dim,))
        self.b_ep = 0.0

    def score_path_noisy_or(self, node_risks: List[float]) -> float:
        r"""
        Calculates Noisy-OR risk aggregation:
            R_path = 1 - \prod_i (1 - R_i)
        Guaranteed to be monotonic and bounded in [0, 1].
        """
        if not node_risks:
            return 0.0
        
        prod_complement = 1.0
        for r in node_risks:
            clamped_r = max(0.0, min(1.0, float(r)))
            prod_complement *= (1.0 - clamped_r)
            
        path_risk = 1.0 - prod_complement
        return round(float(np.clip(path_risk, 0.0, 1.0)), 4)

    def evaluate_path(self, path_nodes: List[str], entity_risks: Dict[str, float]) -> AttackPath:
        """
        Constructs an AttackPath object from a list of nodes and known risk lookup.
        """
        node_risks = [entity_risks.get(node, 0.10) for node in path_nodes]
        path_risk = self.score_path_noisy_or(node_risks)
        
        crit_node = path_nodes[0] if path_nodes else ""
        if path_nodes:
            max_idx = int(np.argmax(node_risks))
            crit_node = path_nodes[max_idx]
            
        return AttackPath(
            path_nodes=list(path_nodes),
            node_risks=node_risks,
            path_risk=path_risk,
            hop_count=max(0, len(path_nodes) - 1),
            critical_node=crit_node
        )

    def score_episode(
        self,
        episode_entities: List[str],
        entity_embeddings: Dict[str, np.ndarray],
        entity_risks: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Computes an episode-level risk score combining pooled GNN node representations
        and individual node risk distributions.
        """
        if not episode_entities:
            return 0.0
            
        # Collect available node embeddings
        valid_embeds = [
            entity_embeddings[e] for e in episode_entities 
            if e in entity_embeddings and entity_embeddings[e] is not None
        ]
        
        if valid_embeds:
            z_ep = np.mean(np.array(valid_embeds), axis=0)
            logit = float(np.dot(z_ep, self.W_ep) + self.b_ep)
            embed_risk = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))
        else:
            embed_risk = 0.50

        # If individual risk scores available, combine with Noisy-OR
        if entity_risks:
            e_risks = [entity_risks.get(e, 0.10) for e in episode_entities]
            path_risk = self.score_path_noisy_or(e_risks)
            # Fused episode risk
            composite = 0.60 * path_risk + 0.40 * embed_risk
        else:
            composite = embed_risk
            
        return round(float(np.clip(composite, 0.0, 1.0)), 4)
