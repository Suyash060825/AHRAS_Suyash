"""AHRAS Explainable AI (XAI) Package"""
from xai.llm_narrator import LLMThreatNarrator, LLMNarrative, get_llm_narrator
from xai.fidelity_ledger import (
    XAIFidelityLedger, XAIFidelityRecord, get_fidelity_ledger, ATTACK_GROUND_TRUTH_FEATURES,
)
from xai.computational_fidelity import (
    ComputationalFidelityEvaluator, FidelityMetrics, ComprehensiveFidelityReport, EventFidelityEvaluation,
)

__all__ = [
    "LLMThreatNarrator", "LLMNarrative", "get_llm_narrator",
    "XAIFidelityLedger", "XAIFidelityRecord", "get_fidelity_ledger", "ATTACK_GROUND_TRUTH_FEATURES",
    "ComputationalFidelityEvaluator", "FidelityMetrics", "ComprehensiveFidelityReport", "EventFidelityEvaluation",
]

