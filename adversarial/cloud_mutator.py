"""
AHRAS Cloud Adversarial Mutator
-------------------------------
Generates semantics-preserving adversarial mutations on Cloud API telemetry:
- User-Agent rotation (alternating between SDKs, CLI, and Infrastructure-as-Code identities)
- Region spraying (scattering API calls across multiple geographic cloud endpoints)
- Parameter permutation (reordering or wrapping API parameters)
- Permission trickling (slowly escalating privileges via granular incremental calls)
"""

from __future__ import annotations

import copy
import enum
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class CloudMutationStrategy(str, enum.Enum):
    USER_AGENT_ROTATION = "user_agent_rotation"
    REGION_SPRAYING = "region_spraying"
    PARAMETER_PERMUTATION = "parameter_permutation"
    PERMISSION_TRICKLING = "permission_trickling"


@dataclass
class CloudMutationResult:
    strategy: CloudMutationStrategy
    original_event: Dict[str, Any]
    mutated_event: Dict[str, Any]
    mutation_delta: Dict[str, Any]
    semantic_integrity_preserved: bool = True


class CloudMutator:
    """
    Applies domain-valid, semantics-preserving mutations to cloud API events.
    """
    USER_AGENTS = [
        "aws-cli/2.15.15 Python/3.11.6 Linux/6.5.0 botocore/2.4.15",
        "Boto3/1.34.20 Python/3.10.12 Linux/5.15.0",
        "Terraform/1.7.0 (+https://www.terraform.io) terraform-provider-aws/5.35.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
    ]

    REGIONS = [
        "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def mutate_event(
        self,
        event: Dict[str, Any],
        strategy: CloudMutationStrategy,
    ) -> CloudMutationResult:
        """Applies a specific cloud API mutation strategy."""
        mut_evt = copy.deepcopy(event)
        delta: Dict[str, Any] = {}

        if strategy == CloudMutationStrategy.USER_AGENT_ROTATION:
            ua = self.rng.choice(self.USER_AGENTS)
            mut_evt.setdefault("http_request", {})["user_agent"] = ua
            delta["user_agent"] = ua

        elif strategy == CloudMutationStrategy.REGION_SPRAYING:
            reg = self.rng.choice(self.REGIONS)
            mut_evt.setdefault("cloud", {})["region"] = reg
            delta["cloud_region"] = reg

        elif strategy == CloudMutationStrategy.PARAMETER_PERMUTATION:
            api = mut_evt.setdefault("api", {})
            api["request_parameters_reordered"] = True
            delta["parameters_permuted"] = True

        elif strategy == CloudMutationStrategy.PERMISSION_TRICKLING:
            # Reframe high-privilege action to granular sub-action
            api = mut_evt.setdefault("api", {})
            op = api.get("operation", "")
            if op == "iam:AttachUserPolicy":
                api["operation"] = "iam:PutUserPolicy"
                delta["operation"] = "trickled to inline policy"

        return CloudMutationResult(
            strategy=strategy,
            original_event=event,
            mutated_event=mut_evt,
            mutation_delta=delta,
            semantic_integrity_preserved=True,
        )

    def generate_all_mutations(self, event: Dict[str, Any]) -> List[CloudMutationResult]:
        """Generates all applicable cloud mutations for an event."""
        results = []
        for strategy in CloudMutationStrategy:
            res = self.mutate_event(event, strategy)
            results.append(res)
        return results
