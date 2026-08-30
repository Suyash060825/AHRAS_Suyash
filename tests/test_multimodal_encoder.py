import pytest
import numpy as np
from detection.multimodal_encoder import (
    ModalityMLP,
    CrossModalAttention,
    MultimodalSecurityEncoder,
    ModalityVectors
)

def test_modality_mlp_shape_and_range():
    mlp = ModalityMLP(in_dim=6, hidden_dim=12, out_dim=8, seed=42)
    x = np.random.randn(6)
    out = mlp.forward(x)
    assert out.shape == (8,)
    assert not np.isnan(out).any()

def test_cross_modal_attention():
    attn = CrossModalAttention(embed_dim=8, num_heads=2, seed=42)
    stack = np.random.randn(4, 8)
    fused, weights = attn.forward(stack)
    assert fused.shape == (8,)
    assert weights.shape == (4, 4)
    # Check softmax sum across attention rows == 1
    assert np.allclose(np.sum(weights, axis=-1), np.ones(4), atol=1e-4)

def test_multimodal_security_encoder():
    encoder = MultimodalSecurityEncoder(embed_dim=8, seed=42)
    evt = {
        "bytes_in": 15000.0,
        "bytes_out": 400.0,
        "packet_count": 120.0,
        "cmd_length": 45.0,
        "path_depth": 3.0,
        "privilege_level": 2.0,
        "in_degree": 4.0,
    }
    vectors = encoder.encode(evt)
    assert isinstance(vectors, ModalityVectors)
    assert vectors.network.shape == (8,)
    assert vectors.process.shape == (8,)
    assert vectors.identity.shape == (8,)
    assert vectors.graph.shape == (8,)
    assert vectors.fused.shape == (8,)
    assert not np.isnan(vectors.fused).any()
