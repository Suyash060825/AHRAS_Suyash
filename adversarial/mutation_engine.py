"""
AHRAS Adversarial Mutation Engine
---------------------------------
Unified coordinator for multi-modal semantics-preserving adversarial perturbations.
Dispatches events to process, network, and cloud mutators while enforcing
strict domain integrity and protocol RFC validity constraints.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from adversarial.process_mutator import ProcessMutator, ProcessMutationStrategy
from adversarial.traffic_mutator import TrafficMutator, TrafficMutationStrategy
from adversarial.cloud_mutator import CloudMutator, CloudMutationStrategy
from detection_coverage.implementation_catalog import TechniqueImplementation, get_default_catalog

log = logging.getLogger(__name__)


@dataclass
class MutatedVariant:
    """
    An individual mutated variant of a technique vector.
    """
    mutation_id: str
    modality: str                        # process, network, cloud, encrypted_session
    strategy_name: str
    epsilon_budget: float                # 0.00 to 0.20
    original_event: Dict[str, Any]
    mutated_event: Dict[str, Any]
    description: str
    semantic_valid: bool = True


@dataclass
class AdversarialPerturbationBatch:
    """
    A collection of mutated variants across perturbation budgets.
    """
    implementation_id: str
    technique_id: str
    vector_name: str
    modality: str
    baseline_event: Dict[str, Any]
    variants: List[MutatedVariant] = field(default_factory=list)


class MutationEngine:
    """
    Unified multi-modal adversarial mutation harness.
    """
    def __init__(self, seed: int = 42) -> None:
        self.process_mutator = ProcessMutator(seed=seed)
        self.traffic_mutator = TrafficMutator(seed=seed)
        self.cloud_mutator = CloudMutator(seed=seed)

    def validate_semantic_integrity(self, event: Dict[str, Any]) -> bool:
        """
        Validates domain constraints:
        - Packets >= 1
        - Durations > 0.0
        - Ports within [1, 65535]
        - Command line non-empty if process activity
        """
        ocsf_cls = event.get("ocsf_class", "")

        if ocsf_cls == "network_activity":
            traf = event.get("traffic", {})
            if "packets" in traf and traf["packets"] < 1:
                return False
            if "duration_sec" in traf and traf["duration_sec"] <= 0.0:
                return False
            dst_p = event.get("dst_endpoint", {}).get("port")
            if dst_p is not None and not (1 <= dst_p <= 65535):
                return False

        elif ocsf_cls == "process_activity":
            actor_proc = event.get("actor", {}).get("process", {})
            raw_proc = event.get("process", {})
            cmd = actor_proc.get("cmd_line") or raw_proc.get("cmd") or ""
            if not cmd:
                return False

        return True

    def generate_perturbations_for_implementation(
        self,
        impl: TechniqueImplementation,
        epsilon_levels: Optional[List[float]] = None,
    ) -> AdversarialPerturbationBatch:
        """
        Generates semantics-preserving mutations for a catalog implementation vector across epsilon levels.
        """
        if epsilon_levels is None:
            epsilon_levels = [0.00, 0.05, 0.10, 0.15, 0.20]

        batch = AdversarialPerturbationBatch(
            implementation_id=impl.implementation_id,
            technique_id=impl.technique_id,
            vector_name=impl.vector_name,
            modality=impl.execution_modality,
            baseline_event=copy.deepcopy(impl.sample_event),
        )

        sample = impl.sample_event or {}
        ocsf_cls = sample.get("ocsf_class", "")

        # Always include baseline as epsilon=0.00
        batch.variants.append(MutatedVariant(
            mutation_id=f"{impl.implementation_id}-EPS-00",
            modality=impl.execution_modality,
            strategy_name="baseline_identity",
            epsilon_budget=0.00,
            original_event=sample,
            mutated_event=copy.deepcopy(sample),
            description="Unperturbed baseline event",
            semantic_valid=True,
        ))

        # 1. Process Activity
        if ocsf_cls == "process_activity":
            strategies = [
                (ProcessMutationStrategy.CASE_ALTERNATION, "Case Alternation Obfuscation"),
                (ProcessMutationStrategy.CARET_INSERTION, "CMD Escape Caret Obfuscation"),
                (ProcessMutationStrategy.QUOTE_INSERTION, "Command Token Quote Obfuscation"),
                (ProcessMutationStrategy.WHITESPACE_PADDING, "Whitespace Padding"),
                (ProcessMutationStrategy.PATH_VARIATION, "Absolute/Relative Path Variation"),
                (ProcessMutationStrategy.FLAG_REORDERING, "CLI Flag Reordering"),
                (ProcessMutationStrategy.ALIAS_SUBSTITUTION, "Binary Alias Substitution"),
            ]
            for idx, (strat, desc) in enumerate(strategies):
                eps = epsilon_levels[min(idx + 1, len(epsilon_levels) - 1)]
                res = self.process_mutator.mutate_event(sample, strat)
                if self.validate_semantic_integrity(res.mutated_event):
                    batch.variants.append(MutatedVariant(
                        mutation_id=f"{impl.implementation_id}-MUT-{idx+1:02d}",
                        modality="process",
                        strategy_name=strat.value,
                        epsilon_budget=eps,
                        original_event=sample,
                        mutated_event=res.mutated_event,
                        description=desc,
                        semantic_valid=True,
                    ))

        # 2. Network Activity & Encrypted Sessions
        elif ocsf_cls in ("network_activity", "encrypted_session"):
            strategies = [
                (TrafficMutationStrategy.PACKET_PADDING, "Packet Size Jitter & Padding"),
                (TrafficMutationStrategy.TIMING_JITTER, "Inter-Arrival Timing Jitter"),
                (TrafficMutationStrategy.DURATION_DILATION, "Session Duration Dilation (PPS Throttling)"),
                (TrafficMutationStrategy.PORT_VARIATION, "Port Shift / Non-Standard Port"),
            ]
            for idx, (strat, desc) in enumerate(strategies):
                eps = epsilon_levels[min(idx + 1, len(epsilon_levels) - 1)]
                res = self.traffic_mutator.mutate_event(sample, strat, intensity=eps * 5.0)
                if self.validate_semantic_integrity(res.mutated_event):
                    batch.variants.append(MutatedVariant(
                        mutation_id=f"{impl.implementation_id}-MUT-{idx+1:02d}",
                        modality="network",
                        strategy_name=strat.value,
                        epsilon_budget=eps,
                        original_event=sample,
                        mutated_event=res.mutated_event,
                        description=desc,
                        semantic_valid=True,
                    ))

        # 3. Cloud API Activity
        elif ocsf_cls == "cloud_api":
            strategies = [
                (CloudMutationStrategy.USER_AGENT_ROTATION, "User-Agent Header Rotation"),
                (CloudMutationStrategy.REGION_SPRAYING, "Multi-Region Spraying"),
                (CloudMutationStrategy.PERMISSION_TRICKLING, "Granular Permission Trickling"),
            ]
            for idx, (strat, desc) in enumerate(strategies):
                eps = epsilon_levels[min(idx + 1, len(epsilon_levels) - 1)]
                res = self.cloud_mutator.mutate_event(sample, strat)
                if self.validate_semantic_integrity(res.mutated_event):
                    batch.variants.append(MutatedVariant(
                        mutation_id=f"{impl.implementation_id}-MUT-{idx+1:02d}",
                        modality="cloud",
                        strategy_name=strat.value,
                        epsilon_budget=eps,
                        original_event=sample,
                        mutated_event=res.mutated_event,
                        description=desc,
                        semantic_valid=True,
                    ))

        return batch
