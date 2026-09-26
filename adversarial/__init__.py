"""
AHRAS Semantic Adversarial Mutation & Evasion Robustness Engine
--------------------------------------------------------------
Evaluates intrusion detection resilience against semantics-preserving
adversarial perturbations across host process, network traffic, and cloud API vectors.
"""

from adversarial.process_mutator import ProcessMutator, ProcessMutationStrategy
from adversarial.traffic_mutator import TrafficMutator, TrafficMutationStrategy
from adversarial.cloud_mutator import CloudMutator, CloudMutationStrategy
from adversarial.mutation_engine import MutationEngine, AdversarialPerturbationBatch
from adversarial.evasion_evaluator import (
    EvasionEvaluator,
    DetectorRobustnessScore,
    EvasionEvaluationReport,
)

__all__ = [
    "ProcessMutator",
    "ProcessMutationStrategy",
    "TrafficMutator",
    "TrafficMutationStrategy",
    "CloudMutator",
    "CloudMutationStrategy",
    "MutationEngine",
    "AdversarialPerturbationBatch",
    "EvasionEvaluator",
    "DetectorRobustnessScore",
    "EvasionEvaluationReport",
]
