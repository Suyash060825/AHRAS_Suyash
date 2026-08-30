import pytest
import numpy as np
from adaptive_learning.weight_learner import (
    EvidenceQualityEngine,
    EvidenceIndependenceController,
    MultiMemoryReplayBuffer,
    FeedbackSample
)

def test_evidence_quality_freshness_and_scoring():
    engine = EvidenceQualityEngine(half_life_sec=300.0)
    
    # Fresh evidence (age = 0)
    q_fresh = engine.evaluate_quality("signature", age_seconds=0.0)
    assert q_fresh.freshness == 1.0
    assert q_fresh.quality_score > 0.80
    
    # Stale evidence (age = 600s, 2 half-lives)
    q_stale = engine.evaluate_quality("signature", age_seconds=600.0)
    assert q_stale.freshness < 0.30
    assert q_stale.quality_score < q_fresh.quality_score

def test_evidence_independence_decorrelation():
    ctrl = EvidenceIndependenceController(n_sources=7)
    weights = {
        "w_sig": 0.30, "w_ml": 0.20, "w_trust": 0.10,
        "w_hist": 0.10, "w_graph": 0.10, "w_fore": 0.10, "w_ti": 0.10
    }
    
    # Set artificial high correlation between w_sig (idx 0) and w_ml (idx 1)
    ctrl.corr_matrix[0, 1] = 0.90
    ctrl.corr_matrix[1, 0] = 0.90
    
    decorr = ctrl.decorrelate_weights(weights)
    assert pytest.approx(sum(decorr.values()), 0.01) == 1.0
    # Overlapping weights should receive a discount relative to uncorrelated weights
    assert decorr["w_sig"] < weights["w_sig"]

def test_multi_memory_replay_buffer():
    buf = MultiMemoryReplayBuffer()
    
    s_benign = FeedbackSample(src_ip="10.0.0.1", label=0, components={"sig": 0.1}, predicted_risk=0.1)
    s_attack = FeedbackSample(src_ip="10.0.0.2", label=1, components={"sig": 0.9}, predicted_risk=0.85)
    s_hard_neg = FeedbackSample(src_ip="10.0.0.3", label=0, components={"sig": 0.7}, predicted_risk=0.65)
    
    buf.add_sample(s_benign, loss=0.02)
    buf.add_sample(s_attack, loss=0.10)
    buf.add_sample(s_hard_neg, loss=0.45)
    
    stats = buf.get_stats()
    assert stats["recent_count"] == 3
    assert stats["attack_count"] == 1
    assert stats["hard_negative_count"] == 1
    
    batch = buf.sample_balanced_batch(batch_size=2)
    assert len(batch) <= 2
