"""
AHRAS Tier Cost Model
---------------------
Defines the 5-Tier computational execution hierarchy:
Tier 0: O(1) stateless filter / hash lookups (microseconds)
Tier 1: Streaming sketches / lightweight statistical tests (< 1ms)
Tier 2: Tree-based anomaly / shallow ML (1-5ms)
Tier 3: Deep learning / GNN / representation models (10-50ms)
Tier 4: Full causal XAI / attack graph expansion (asynchronous / forensic)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class ExecutionTier(int, enum.Enum):
    TIER_0_STATELESS = 0
    TIER_1_STREAMING_SKETCH = 1
    TIER_2_SHALLOW_ML = 2
    TIER_3_DEEP_GNN = 3
    TIER_4_CAUSAL_FORENSIC = 4


@dataclass
class TierSpecification:
    """
    Cost, latency, and capability profile of an execution tier.
    """
    tier: ExecutionTier
    name: str
    description: str
    nominal_latency_ms: float      # Processing latency in milliseconds
    cpu_cost_factor: float         # Normalized CPU cost (1.0 = baseline light)
    memory_mb: float               # Working set RAM consumed
    max_throughput_eps: float      # Upper bound capacity (events per second)
    detection_capability: float    # Fraction of complex/zero-day attacks resolvable


DEFAULT_TIER_SPECS: Dict[ExecutionTier, TierSpecification] = {
    ExecutionTier.TIER_0_STATELESS: TierSpecification(
        tier=ExecutionTier.TIER_0_STATELESS,
        name="Tier 0: Stateless Filter",
        description="O(1) hash table lookups, IP threat intel sets, static drop lists",
        nominal_latency_ms=0.03,
        cpu_cost_factor=0.05,
        memory_mb=0.8,
        max_throughput_eps=33000.0,
        detection_capability=0.35,
    ),
    ExecutionTier.TIER_1_STREAMING_SKETCH: TierSpecification(
        tier=ExecutionTier.TIER_1_STREAMING_SKETCH,
        name="Tier 1: Streaming Sketch",
        description="Count-Min sketch, Welford variance tracking, pre-compiled regex",
        nominal_latency_ms=0.35,
        cpu_cost_factor=0.25,
        memory_mb=2.5,
        max_throughput_eps=8500.0,
        detection_capability=0.60,
    ),
    ExecutionTier.TIER_2_SHALLOW_ML: TierSpecification(
        tier=ExecutionTier.TIER_2_SHALLOW_ML,
        name="Tier 2: Shallow Anomaly ML",
        description="Isolation Forest, One-Class SVM, lightweight decision tree ensemble",
        nominal_latency_ms=2.80,
        cpu_cost_factor=1.00,
        memory_mb=18.0,
        max_throughput_eps=1200.0,
        detection_capability=0.82,
    ),
    ExecutionTier.TIER_3_DEEP_GNN: TierSpecification(
        tier=ExecutionTier.TIER_3_DEEP_GNN,
        name="Tier 3: Deep Representation & GNN",
        description="Temporal GNN message passing, multimodal cross-attention encoder",
        nominal_latency_ms=18.50,
        cpu_cost_factor=4.50,
        memory_mb=65.0,
        max_throughput_eps=180.0,
        detection_capability=0.95,
    ),
    ExecutionTier.TIER_4_CAUSAL_FORENSIC: TierSpecification(
        tier=ExecutionTier.TIER_4_CAUSAL_FORENSIC,
        name="Tier 4: Causal XAI & Attack Graph",
        description="Counterfactual search, multi-hop provenance graph traversal, security twin",
        nominal_latency_ms=85.00,
        cpu_cost_factor=15.00,
        memory_mb=160.0,
        max_throughput_eps=25.0,
        detection_capability=0.99,
    ),
}


class TierCostModel:
    """
    Computes computational resource costs and capacity for execution tiers.
    """
    def __init__(self, specs: Optional[Dict[ExecutionTier, TierSpecification]] = None) -> None:
        self.specs = specs or dict(DEFAULT_TIER_SPECS)

    def get_spec(self, tier: ExecutionTier | int) -> TierSpecification:
        t = ExecutionTier(tier)
        return self.specs[t]

    def estimate_batch_cost(self, tier: ExecutionTier, batch_size: int) -> Dict[str, float]:
        """Estimates total execution time and CPU consumption for a batch of events."""
        spec = self.get_spec(tier)
        total_time_ms = spec.nominal_latency_ms * batch_size
        cpu_work = spec.cpu_cost_factor * batch_size
        return {
            "tier": tier.value,
            "total_latency_ms": round(total_time_ms, 2),
            "cpu_work_units": round(cpu_work, 2),
            "memory_mb": spec.memory_mb,
        }
