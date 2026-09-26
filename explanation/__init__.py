"""
AHRAS Trustworthy Explanation Stability & Counterfactual Verification Package
-----------------------------------------------------------------------------
Audits feature attribution stability, verifies counterfactual actionability and sparsity,
evaluates faithfulness (Sufficiency, Comprehensiveness, Monotonicity),
and computes cross-explainer consensus (EDI).
"""

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

__all__ = [
    "ExplanationStabilityAuditor",
    "StabilityMetrics",
    "CounterfactualVerifier",
    "CounterfactualVerificationResult",
    "FaithfulnessEvaluator",
    "FaithfulnessMetrics",
    "ExplainerConsensusEngine",
    "ConsensusReport",
]
