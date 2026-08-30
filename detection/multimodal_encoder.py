from __future__ import annotations
"""
AHRAS Module — Multimodal Security Representation & Cross-Modal Attention Encoder
----------------------------------------------------------------------------------
Splits raw heterogeneous telemetry events into typed modality representations:
  1. Network Telemetry (bytes/s, packet rates, port profiles, protocol flags)
  2. Process & Execution (command line tokens, process tree depth, CPU/memory signatures)
  3. Identity & Authentication (user privilege tier, auth failure counts, session duration)
  4. Graph & Relational Context (node degree, centrality proxy, local clustering)

Cross-Modal Attention:
  Computes scaled dot-product cross-modal attention across typed embeddings to generate
  a unified joint security representation z_security in R^D.
"""

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
        # Xavier / He initialization
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for single sample or batch.
        x: shape (in_dim,) or (batch_size, in_dim)
        """
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
        d_k = self.head_dim
        
        Q = modality_embeddings @ self.W_q  # (M, embed_dim)
        K = modality_embeddings @ self.W_k  # (M, embed_dim)
        V = modality_embeddings @ self.W_v  # (M, embed_dim)
        
        # Reshape to (num_heads, M, head_dim)
        Q_h = Q.reshape(M, self.num_heads, d_k).swapaxes(0, 1)
        K_h = K.reshape(M, self.num_heads, d_k).swapaxes(0, 1)
        V_h = V.reshape(M, self.num_heads, d_k).swapaxes(0, 1)
        
        # Scaled dot-product attention per head
        # scores: (num_heads, M, M)
        scores = (Q_h @ K_h.swapaxes(1, 2)) / np.sqrt(float(d_k))
        # Softmax over last dimension
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_h = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
        
        # Context per head: (num_heads, M, head_dim)
        out_h = attn_h @ V_h
        # Recombine heads: (M, embed_dim)
        out = out_h.swapaxes(0, 1).reshape(M, self.embed_dim) @ self.W_o
        
        # Mean pool across modalities to obtain final joint embedding
        fused = np.mean(out, axis=0)
        # Average attention across heads for interpretability
        avg_attn = np.mean(attn_h, axis=0)
        
        return fused, avg_attn


@dataclass
class ModalityVectors:
    network: np.ndarray
    process: np.ndarray
    identity: np.ndarray
    graph: np.ndarray
    fused: np.ndarray
    attention_matrix: np.ndarray


class MultimodalSecurityEncoder:
    """
    Extracts 4 typed feature modalities from raw or normalized security events,
    encodes them through modality-specific MLPs, and fuses them via cross-modal attention.
    """

    def __init__(self, embed_dim: int = 8, seed: int = 42):
        self.embed_dim = embed_dim
        # Network modality: 6 features (bytes_in, bytes_out, pkts_in, pkts_out, port_entropy, duration)
        self.enc_net = ModalityMLP(in_dim=6, hidden_dim=12, out_dim=embed_dim, seed=seed)
        # Process modality: 4 features (cmd_length, path_depth, is_elevated, cpu_pct)
        self.enc_proc = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=embed_dim, seed=seed + 1)
        # Identity modality: 4 features (privilege_level, failed_auth_count, session_age_sec, concurrent_logins)
        self.enc_id = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=embed_dim, seed=seed + 2)
        # Graph modality: 4 features (in_degree, out_degree, neighbor_anomaly_mean, local_clustering)
        self.enc_graph = ModalityMLP(in_dim=4, hidden_dim=8, out_dim=embed_dim, seed=seed + 3)
        
        self.attn = CrossModalAttention(embed_dim=embed_dim, num_heads=2, seed=seed + 4)

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

    def encode(self, evt: Dict[str, Any]) -> ModalityVectors:
        """Encodes an event into unimodal representations and fused joint embedding."""
        x_net, x_proc, x_id, x_graph = self.extract_modality_raw_features(evt)
        
        z_net = self.enc_net.forward(x_net)
        z_proc = self.enc_proc.forward(x_proc)
        z_id = self.enc_id.forward(x_id)
        z_graph = self.enc_graph.forward(x_graph)
        
        modality_stack = np.stack([z_net, z_proc, z_id, z_graph], axis=0)
        fused, attn_mat = self.attn.forward(modality_stack)
        
        return ModalityVectors(
            network=z_net,
            process=z_proc,
            identity=z_id,
            graph=z_graph,
            fused=fused,
            attention_matrix=attn_mat
        )
