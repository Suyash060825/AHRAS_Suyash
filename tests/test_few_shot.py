"""
Tests for Few-Shot Novel Attack Adaptation Engine (Section 17 / Research Frontier P1)
--------------------------------------------------------------------------------------
Validates:
  1. Discovery of novel cluster and analyst confirmation workflow.
  2. Prototypical centroid adaptation across 1-shot, 5-shot, 10-shot, 25-shot.
  3. Regularized linear head adaptation with L2 penalty.
  4. Continual replay update with learning rate decay.
  5. Old attack retention and benign FPR safety checks.
  6. Lifecycle progression: DISCOVERED -> ANALYST_CONFIRMED -> ADAPTED -> VALIDATED -> SHADOW -> PROMOTED.
"""

import pytest
import numpy as np
from adaptive_learning.few_shot import (
    FewShotAttackAdapter,
    AdaptationStrategy,
    AdaptationLifecycleStage,
    NovelAttackCluster,
    AdaptationResult,
)


@pytest.fixture
def seeded_adapter():
    rng = np.random.default_rng(42)
    adapter = FewShotAttackAdapter(embed_dim=14, temperature=0.1)
    
    # 50 benign samples centered near 0
    benign_samples = rng.normal(0.0, 0.05, size=(50, 14))
    # 50 known attack samples centered near 0.7
    attack_samples = rng.normal(0.7, 0.05, size=(50, 14))
    
    adapter.seed_historical_holdout(benign_samples, attack_samples)
    return adapter


def test_cluster_discovery_and_analyst_confirmation(seeded_adapter):
    exemplars = [[0.85] * 14, [0.88] * 14, [0.83] * 14]
    cluster = seeded_adapter.register_unlabeled_cluster(
        exemplar_features=exemplars,
        event_ids=["evt-1", "evt-2", "evt-3"],
        ood_score=0.92,
        uncertainty=0.81,
    )
    assert cluster.status == AdaptationLifecycleStage.DISCOVERED
    assert cluster.mean_ood_score == 0.92
    assert len(cluster.seed_event_ids) == 3

    confirmed = seeded_adapter.confirm_analyst_label(
        cluster_id=cluster.cluster_id,
        analyst_label="ZeroDay-Ransomware-Variant-X",
    )
    assert confirmed.status == AdaptationLifecycleStage.ANALYST_CONFIRMED
    assert confirmed.analyst_label == "ZeroDay-Ransomware-Variant-X"
    assert confirmed.confirmed_at is not None


@pytest.mark.parametrize("shots", [1, 5, 10, 25])
def test_prototypical_adaptation_shots(seeded_adapter, shots):
    rng = np.random.default_rng(100 + shots)
    # Target novel attack centered at 1.4
    novel_samples = rng.normal(1.4, 0.05, size=(shots, 14))

    res = seeded_adapter.adapt_few_shot(
        attack_name="ZeroDay-Alpha",
        support_samples=novel_samples,
        strategy=AdaptationStrategy.PROTOTYPE_CENTROID,
    )

    assert res.k_shots == shots
    assert res.attack_name == "ZeroDay-Alpha"
    assert res.adaptation_latency_ms > 0.0
    assert res.validation_new_attack_recall >= 0.80
    assert res.validation_old_attack_retention >= 0.90
    assert res.validation_fpr <= 0.05
    assert res.is_safe_to_shadow is True

    # Test inference on an unseen instance of this attack
    test_probe = novel_samples[0] + rng.normal(0, 0.02, size=14)
    pred_label, conf = seeded_adapter.predict_sample(test_probe)
    assert pred_label == "ZeroDay-Alpha"
    assert conf > 0.50


def test_regularized_linear_head_adaptation(seeded_adapter):
    rng = np.random.default_rng(42)
    novel_samples = rng.normal(1.2, 0.04, size=(10, 14))

    res = seeded_adapter.adapt_few_shot(
        attack_name="ZeroDay-Beta",
        support_samples=novel_samples,
        strategy=AdaptationStrategy.REGULARIZED_HEAD,
    )
    assert res.strategy == AdaptationStrategy.REGULARIZED_HEAD
    assert res.is_safe_to_shadow is True
    assert "weight_l2" in res.weights_summary

    # Predict with linear head
    test_probe = novel_samples[0] + rng.normal(0, 0.01, size=14)
    pred_label, conf = seeded_adapter.predict_sample(test_probe)
    assert pred_label == "ZeroDay-Beta"
    assert conf > 0.50


def test_continual_replay_update_adaptation(seeded_adapter):
    rng = np.random.default_rng(42)
    novel_samples = rng.normal(0.95, 0.05, size=(5, 14))

    res = seeded_adapter.adapt_few_shot(
        attack_name="ZeroDay-Gamma",
        support_samples=novel_samples,
        strategy=AdaptationStrategy.CONTINUAL_REPLAY_UPDATE,
    )
    assert res.strategy == AdaptationStrategy.CONTINUAL_REPLAY_UPDATE
    assert res.validation_old_attack_retention >= 0.90


def test_shadow_and_production_promotion_lifecycle(seeded_adapter):
    rng = np.random.default_rng(42)
    novel_samples = rng.normal(1.5, 0.05, size=(5, 14))

    res = seeded_adapter.adapt_few_shot(
        attack_name="ZeroDay-Delta",
        support_samples=novel_samples,
        strategy=AdaptationStrategy.PROTOTYPE_CENTROID,
    )
    assert res.lifecycle_stage == AdaptationLifecycleStage.VALIDATED

    # Cannot promote directly to production without shadow
    assert seeded_adapter.promote_to_production(res.adaptation_id) is False

    # Promote to shadow
    assert seeded_adapter.promote_to_shadow(res.adaptation_id) is True
    updated = seeded_adapter.get_adaptation(res.adaptation_id)
    assert updated.lifecycle_stage == AdaptationLifecycleStage.SHADOW_DEPLOYED

    # Now promote to production
    assert seeded_adapter.promote_to_production(res.adaptation_id) is True
    updated = seeded_adapter.get_adaptation(res.adaptation_id)
    assert updated.lifecycle_stage == AdaptationLifecycleStage.PROMOTED
