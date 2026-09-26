"""
Tests for AHRAS Trustworthy Explanation Stability & Counterfactual Verification (Frontier G / EXP-28)
------------------------------------------------------------------------------------------------------
Validates:
  1. ExplanationStabilityAuditor Jaccard stability, rank correlation, and Lipschitz bounds.
  2. CounterfactualVerifier target validity, actionability, immutability, and boundary checks.
  3. FaithfulnessEvaluator Sufficiency, Comprehensiveness, and Monotonicity under feature masking.
  4. ExplainerConsensusEngine Explainer Disagreement Index (EDI) and unanimous feature extraction.
"""

import pytest
import numpy as np

from explanation.stability_auditor import (
    ExplanationStabilityAuditor,
    StabilityMetrics,
)
from explanation.counterfactual_verifier import (
    CounterfactualVerifier,
    CounterfactualVerificationResult,
)
from explanation.faithfulness_evaluator import (
    FaithfulnessEvaluator,
    FaithfulnessMetrics,
)
from explanation.cross_model_consensus import (
    ExplainerConsensusEngine,
    ConsensusReport,
)


class TestExplanationStabilityAuditor:
    def test_stable_attribution(self):
        auditor = ExplanationStabilityAuditor(noise_std=0.02, n_perturbations=30)
        # Linear attribution function: phi(x) = W * x
        W = np.array([10.0, 5.0, 3.0, 1.0, 0.5, 0.1, 0.05, 0.01])
        base_x = np.ones(8)

        def attr_func(x):
            return W * x

        metrics = auditor.evaluate_attribution_stability(base_x, attr_func, top_k=3)
        assert metrics.is_stable
        assert metrics.mean_jaccard_similarity >= 0.85
        assert metrics.spearman_rho >= 0.80
        assert metrics.kendall_tau >= 0.70
        assert metrics.lipschitz_continuity > 0.0

    def test_unstable_chaotic_attribution(self):
        auditor = ExplanationStabilityAuditor(noise_std=0.10, n_perturbations=30)
        base_x = np.ones(8)

        # Chaotic attribution sensitive to small noise
        def chaotic_attr(x):
            return np.sin(x * 100.0)

        metrics = auditor.evaluate_attribution_stability(base_x, chaotic_attr, top_k=3)
        # High noise on high frequency sin should yield low Jaccard stability
        assert metrics.mean_jaccard_similarity < 0.80
        assert not metrics.is_stable


class TestCounterfactualVerifier:
    def test_valid_actionable_counterfactual(self):
        verifier = CounterfactualVerifier(
            immutable_feature_indices={0, 1},  # feature 0 and 1 are immutable
            feature_bounds={2: (0.0, 100.0), 3: (0.0, 1.0)},
            feature_names=["src_ip", "protocol", "failed_logins", "is_admin"],
        )
        x_fact = np.array([192.0, 6.0, 85.0, 1.0])
        x_cf = np.array([192.0, 6.0, 5.0, 1.0])  # Modified feature 2 (mutable and within bounds)

        def predict_risk(x):
            return float(x[2] / 100.0)  # Risk = failed_logins / 100

        res = verifier.verify_counterfactual(x_fact, x_cf, predict_risk, target_risk_threshold=0.20)
        assert res.is_valid
        assert res.is_actionable
        assert res.l0_sparsity == 1
        assert res.risk_reduction == pytest.approx(0.80, 0.01)
        assert len(res.immutable_feature_violations) == 0
        assert len(res.out_of_bounds_violations) == 0

    def test_immutable_feature_violation(self):
        verifier = CounterfactualVerifier(
            immutable_feature_indices={0},
            feature_names=["src_ip", "packet_rate"],
        )
        x_fact = np.array([10.0, 500.0])
        x_cf = np.array([20.0, 500.0])  # Illegally attempted to alter immutable src_ip

        def predict_risk(x):
            return 0.1

        res = verifier.verify_counterfactual(x_fact, x_cf, predict_risk)
        assert not res.is_actionable
        assert "src_ip" in res.immutable_feature_violations

    def test_out_of_bounds_violation(self):
        verifier = CounterfactualVerifier(
            feature_bounds={1: (0.0, 100.0)},
            feature_names=["proc_id", "cpu_pct"],
        )
        x_fact = np.array([1000.0, 95.0])
        x_cf = np.array([1000.0, -15.0])  # Negative CPU percentage is physically invalid

        def predict_risk(x):
            return 0.1

        res = verifier.verify_counterfactual(x_fact, x_cf, predict_risk)
        assert not res.is_actionable
        assert len(res.out_of_bounds_violations) == 1


class TestFaithfulnessEvaluator:
    def test_eval_faithfulness(self):
        evaluator = FaithfulnessEvaluator(baseline_val=0.0)
        # Linear risk function: R(x) = sum(w * x)
        w = np.array([0.40, 0.30, 0.20, 0.05, 0.05])
        x = np.ones(5)
        attrs = w * x  # Exact attributions

        def predict_risk(vec):
            return float(np.sum(w * vec))

        metrics = evaluator.evaluate_faithfulness(x, attrs, predict_risk, top_k=3)
        assert metrics.is_faithful
        assert metrics.sufficiency_score >= 0.85
        assert metrics.comprehensiveness_score >= 0.80
        assert metrics.monotonicity_score == 1.0


class TestExplainerConsensusEngine:
    def test_high_consensus_between_explainers(self):
        engine = ExplainerConsensusEngine(majority_threshold=0.60)
        # 3 explainers agreeing on top features (0, 1, 2)
        attrs = {
            "causal_dag": np.array([0.9, 0.8, 0.7, 0.2, 0.1]),
            "counterfactual": np.array([0.85, 0.75, 0.65, 0.1, 0.05]),
            "integrated_gradients": np.array([0.95, 0.88, 0.72, 0.3, 0.15]),
        }
        report = engine.evaluate_consensus(attrs, top_k=3)
        assert report.high_consensus
        assert report.explainer_disagreement_index < 0.15
        assert set(report.unanimous_features) == {0, 1, 2}

    def test_divergent_explainers(self):
        engine = ExplainerConsensusEngine()
        attrs = {
            "method_a": np.array([1.0, 0.0, 0.0, 0.0]),
            "method_b": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        report = engine.evaluate_consensus(attrs, top_k=1)
        assert not report.high_consensus
        assert report.explainer_disagreement_index == 1.0
        assert len(report.unanimous_features) == 0
