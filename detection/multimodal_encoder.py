from __future__ import annotations
"""
AHRAS Module — Multimodal Security Representation & Temporal Attention Encoder
--------------------------------------------------------------------------------
Splits raw heterogeneous telemetry events into typed modality representations:
  1. Network Telemetry (bytes/s, packet rates, port profiles, protocol flags)
  2. Process & Execution (command line tokens, process tree depth, CPU/memory signatures)
  3. Identity & Authentication (user privilege tier, auth failure counts, session duration)
  4. Graph & Relational Context (node degree, centrality proxy, local clustering)

Cross-Modal & Temporal Attention Hierarchy:
  1. Modality Encoders: x_m -> z_m in R^D
  2. Cross-Modal Attention: Scaled dot-product across active modalities -> z_t in R^D
  3. Temporal Attention: alpha_t = softmax(score(z_t)) over sliding window T -> z_episode = sum_t alpha_t z_t
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

log = logging.getLogger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return default


class ModalityMLP:
    """Lightweight 2-layer MLP with ReLU activation for encoding an individual modality."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_in = np.asarray(x, dtype=np.float64)
        single = (x_in.ndim == 1)
        if single:
            x_in = x_in[np.newaxis, :]
            
        h = np.maximum(0.0, x_in @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        
        if single:
            return out[0]
        return out


class CrossModalAttention:
    """
    Computes multi-head cross-modal attention across modality embeddings.
    K modalities -> Q, K, V projections -> Unified output representation.
    """

    def __init__(self, embed_dim: int = 8, num_heads: int = 2, seed: int = 42):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        rng = np.random.default_rng(seed)
        self.W_q = rng.normal(0.0, np.sqrt(1.0 / embed_dim), size=(embed_dim, embed_dim))
        self.W_k = rng.normal(0.0, np.sqrt(1.0 / embed_dim), size=(embed_dim, embed_dim))
        self.W_v = rng.normal(0.0, np.sqrt(1.0 / embed_dim), size=(embed_dim, embed_dim))
        self.W_o = rng.normal(0.0, np.sqrt(1.0 / embed_dim), size=(embed_dim, embed_dim))

    def forward(self, modality_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        modality_embeddings: shape (num_modalities, embed_dim)
        Returns:
            fused_vector: shape (embed_dim,)
            attn_weights: shape (num_modalities, num_modalities)
        """
        M = modality_embeddings.shape[0]
        if M == 1:
            # Single modality: direct projection
            fused = modality_embeddings[0] @ self.W_o
            return fused, np.ones((1, 1), dtype=np.float64)

        d_k = self.head_dim
        Q = modality_embeddings @ self.W_q
        K = modality_embeddings @ self.W_k
        V = modality_embeddings @ self.W_v
        
        Q_h = Q.reshape(M, self.num_heads, d_k).swapaxes(0, 1)
        K_h = K.reshape(M, self.num_heads, d_k).swapaxes(0, 1)
        V_h = V.reshape(M, self.num_heads, d_k).swapaxes(0, 1)
        
        scores = (Q_h @ K_h.swapaxes(1, 2)) / np.sqrt(float(d_k))
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_h = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
        
        out_h = attn_h @ V_h
        out = out_h.swapaxes(0, 1).reshape(M, self.embed_dim) @ self.W_o
        
        fused = np.mean(out, axis=0)
        avg_attn = np.mean(attn_h, axis=0)
        
        return fused, avg_attn


class TemporalAttention:
    """
    Computes temporal attention weights alpha_t over a sequence of time-window representations:
        alpha_t = softmax( W_temp * z_t + b_temp )
        z_episode = sum_t alpha_t * z_t
    """

    def __init__(self, embed_dim: int = 8, seed: int = 42):
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        self.w_temp = rng.normal(0.0, np.sqrt(1.0 / embed_dim), size=(embed_dim,))
        self.b_temp = 0.0

    def forward(self, sequence_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        sequence_vectors: shape (T, embed_dim)
        Returns:
            z_episode: shape (embed_dim,)
            temporal_weights: shape (T,)
        """
        seq = np.asarray(sequence_vectors, dtype=np.float64)
        if seq.ndim == 1:
            return seq, np.array([1.0], dtype=np.float64)
        T = seq.shape[0]
        if T == 0:
            return np.zeros(self.embed_dim), np.empty(0)

        # Logit per timestep: s_t = z_t . w_temp + b_temp
        logits = np.dot(seq, self.w_temp) + self.b_temp
        exp_logits = np.exp(logits - np.max(logits))
        alphas = exp_logits / (np.sum(exp_logits) + 1e-12)
        
        # Weighted combination: sum_t alpha_t * z_t
        z_episode = np.sum(seq * alphas[:, np.newaxis], axis=0)
        return z_episode, alphas


@dataclass
class ModalityVectors:
    network: np.ndarray
    process: np.ndarray
    identity: np.ndarray
    graph: np.ndarray
    fused: np.ndarray
    attention_matrix: np.ndarray
    temporal_alphas: Optional[np.ndarray] = None
    episode_embedding: Optional[np.ndarray] = None


class MultimodalSecurityEncoder:
    """
    Hierarchical Multimodal & Temporal Security Encoder:
      - 4 typed modality streams
      - Selective modality masking for ablations
      - Cross-modal attention fusion
      - Sliding-window temporal attention
    """

    def __init__(self, embed_dim: int = 8, seed: int = 42):
        self.embed_dim = embed_dim
        self.enc_net = ModalityMLP(in_dim=6, hidden_dim=12, out_dim=embed_dim, seed=seed)
        self.enc_proc = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=embed_dim, seed=seed + 1)
        self.enc_id = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=embed_dim, seed=seed + 2)
        self.enc_graph = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=embed_dim, seed=seed + 3)
        
        self.attn = CrossModalAttention(embed_dim=embed_dim, num_heads=2, seed=seed + 4)
        self.temporal_attn = TemporalAttention(embed_dim=embed_dim, seed=seed + 5)

    def extract_modality_raw_features(self, evt: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts structured raw numerical vectors from an event dictionary."""
        # 1. Network
        net_raw = np.array([
            _safe_float(evt.get("bytes_in", evt.get("traffic_volume", 0.0))),
            _safe_float(evt.get("bytes_out", 0.0)),
            _safe_float(evt.get("packet_count", evt.get("pkts_in", 0.0))),
            _safe_float(evt.get("pkts_out", 0.0)),
            _safe_float(evt.get("port_entropy", 0.0)),
            _safe_float(evt.get("duration", evt.get("connection_duration", 0.0))),
        ], dtype=np.float64)

        # 2. Process
        proc_raw = np.array([
            _safe_float(evt.get("cmd_length", len(str(evt.get("command", ""))))),
            _safe_float(evt.get("path_depth", 1.0)),
            1.0 if evt.get("is_elevated", False) or evt.get("is_root", False) else 0.0,
            _safe_float(evt.get("cpu_pct", 0.0)),
        ], dtype=np.float64)

        # 3. Identity
        id_raw = np.array([
            _safe_float(evt.get("privilege_level", 1.0)),
            _safe_float(evt.get("failed_auth_count", 0.0)),
            _safe_float(evt.get("session_age_sec", 60.0)),
            _safe_float(evt.get("concurrent_logins", 1.0)),
        ], dtype=np.float64)

        # 4. Graph
        graph_raw = np.array([
            _safe_float(evt.get("in_degree", 1.0)),
            _safe_float(evt.get("out_degree", 1.0)),
            _safe_float(evt.get("neighbor_anomaly_mean", 0.0)),
            _safe_float(evt.get("local_clustering", 0.0)),
        ], dtype=np.float64)

        return net_raw, proc_raw, id_raw, graph_raw

    def encode(
        self,
        evt: Dict[str, Any],
        active_modalities: Optional[Set[str]] = None
    ) -> ModalityVectors:
        """
        Encodes an event into active unimodal representations and fused joint embedding.
        active_modalities: Optional set e.g. {"network", "process"} for rigorous modality ablations.
        """
        allowed = active_modalities or {"network", "process", "identity", "graph"}
        x_net, x_proc, x_id, x_graph = self.extract_modality_raw_features(evt)
        
        z_net = self.enc_net.forward(x_net)
        z_proc = self.enc_proc.forward(x_proc)
        z_id = self.enc_id.forward(x_id)
        z_graph = self.enc_graph.forward(x_graph)
        
        active_stack = []
        if "network" in allowed:
            active_stack.append(z_net)
        if "process" in allowed:
            active_stack.append(z_proc)
        if "identity" in allowed:
            active_stack.append(z_id)
        if "graph" in allowed:
            active_stack.append(z_graph)

        if not active_stack:
            active_stack.append(z_net)

        modality_stack = np.stack(active_stack, axis=0)
        fused, attn_mat = self.attn.forward(modality_stack)
        
        return ModalityVectors(
            network=z_net,
            process=z_proc,
            identity=z_id,
            graph=z_graph,
            fused=fused,
            attention_matrix=attn_mat,
        )

    def encode_sequence(
        self,
        events: List[Dict[str, Any]],
        active_modalities: Optional[Set[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[ModalityVectors]]:
        """
        Encodes a chronological sequence of events into per-event vectors and pooled temporal episode vector:
        Returns: (z_episode, temporal_alphas, per_event_vectors)
        """
        per_event = [self.encode(e, active_modalities=active_modalities) for e in events]
        fused_stack = np.stack([v.fused for v in per_event], axis=0) if per_event else np.zeros((1, self.embed_dim))
        z_ep, alphas = self.temporal_attn.forward(fused_stack)
        
        for idx, v in enumerate(per_event):
            v.temporal_alphas = alphas
            v.episode_embedding = z_ep
            
        return z_ep, alphas, per_event
