import hashlib
import time
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AttackPathPrediction:
    risk_energy: float        # 0.0 to 1.0
    developing: bool          # True if active progression is detected
    stage_estimate: str       # Kill-chain phase estimate
    kill_chain_stages: list   # List of stages

class TemporalGNN:
    """
    Deterministic Temporal Graph Relational Feature Extractor & Directional Drift Engine.
    Uses exponential time-decay on edges to surface multi-stage APT movement
    before the final hop lands. Node embeddings are deterministically initialized via
    hash projection and intermediate embeddings drift toward the source attacker node
    proportional to the temporal edge weight.
    """
    def __init__(self, decay_rate: float = 0.01, seed: int = 42):
        self.decay_rate = decay_rate
        self.seed = seed
        
        # Maps node_id -> embedding (1D numpy array)
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # Adjacency tracking: (src, dst) -> (weight, last_timestamp)
        self.edges: Dict[Tuple[str, str], Tuple[float, float]] = {}
        
        # Default embedding dimension
        self.dim = 32
        
    def _get_embedding(self, node_id: str) -> np.ndarray:
        if node_id not in self.node_embeddings:
            # Deterministic initialization based on SHA-256 hash of node_id and base seed
            h = hashlib.sha256(f"{self.seed}:{node_id}".encode("utf-8")).digest()
            seed_int = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed_int)
            vec = rng.normal(0.0, 0.1, size=self.dim)
            norm = np.linalg.norm(vec)
            self.node_embeddings[node_id] = vec / (norm + 1e-9)
        return self.node_embeddings[node_id]

    def _propagate(self, src: str, dst: str, weight: float) -> float:
        """
        Message passing step: update dst embedding based on src embedding.
        Swapping in a learned deep TGNN model can replace this forward pass.
        """
        emb_src = self._get_embedding(src)
        emb_dst = self._get_embedding(dst)
        
        # Target node drifts towards source node proportional to edge weight
        drift = weight * (emb_src - emb_dst)
        new_emb = emb_dst + 0.2 * drift
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb = new_emb / norm
            
        self.node_embeddings[dst] = new_emb
        
        # Energy is the cosine similarity between the drifted nodes
        energy = float(np.dot(emb_src, new_emb) / (np.linalg.norm(emb_src) * np.linalg.norm(new_emb) + 1e-9))
        return max(0.0, energy)

    def record_interaction(self, src: str, dst: str, timestamp: float, severity: float = 1.0) -> AttackPathPrediction:
        """
        Record a temporal edge and return the updated risk energy prediction.
        """
        current_time = timestamp
        edge_key = (src, dst)
        
        if edge_key in self.edges:
            old_weight, last_time = self.edges[edge_key]
            # Exponential time decay based on elapsed seconds
            dt = max(0.0, current_time - last_time)
            decayed_weight = old_weight * np.exp(-self.decay_rate * dt)
            new_weight = decayed_weight + severity
        else:
            new_weight = severity
            
        self.edges[edge_key] = (new_weight, current_time)
        
        # Propagate the embedding
        energy = self._propagate(src, dst, new_weight)
        
        # Map energy to kill-chain stages
        developing = energy > 0.4
        
        if energy > 0.8:
            stage = "ACTION_ON_OBJECTIVES"
            chain = ["RECON", "INITIAL_ACCESS", "LATERAL_MOVEMENT", "IMPACT"]
        elif energy > 0.6:
            stage = "LATERAL_MOVEMENT"
            chain = ["RECON", "INITIAL_ACCESS", "LATERAL_MOVEMENT"]
        elif energy > 0.4:
            stage = "INITIAL_ACCESS"
            chain = ["RECON", "INITIAL_ACCESS"]
        else:
            stage = "RECONNAISSANCE"
            chain = ["RECON"]
            
        return AttackPathPrediction(
            risk_energy=round(energy, 3),
            developing=developing,
            stage_estimate=stage,
            kill_chain_stages=chain
        )

# Singleton
_tgnn_instance = None
def get_tgnn() -> TemporalGNN:
    global _tgnn_instance
    if _tgnn_instance is None:
        _tgnn_instance = TemporalGNN()
    return _tgnn_instance
