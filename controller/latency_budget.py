"""
AHRAS Latency Budget Manager
----------------------------
Monitors processing latency against strict SLA deadlines (e.g. P99 < 5ms inline),
enforcing dynamic degradation and early exits when budget is exhausted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from controller.cost_model import ExecutionTier, TierSpecification, TierCostModel


@dataclass
class LatencyBudgetPolicy:
    """
    SLA constraints for event inspection.
    """
    max_sla_deadline_ms: float = 10.0      # Maximum allowable end-to-end latency
    target_p99_sla_ms: float = 5.0         # Target P99 latency boundary
    enforce_strict_cutoff: bool = True     # True to reject tiers that would exceed deadline


@dataclass
class LatencyDistributionMetrics:
    """
    Statistical latency distribution across processed events.
    """
    total_events: int
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    sla_violations_count: int
    sla_violation_rate_pct: float


class LatencyBudgetManager:
    """
    Tracks elapsed latency and filters feasible execution tiers against budget.
    """
    def __init__(
        self,
        policy: Optional[LatencyBudgetPolicy] = None,
        cost_model: Optional[TierCostModel] = None,
    ) -> None:
        self.policy = policy or LatencyBudgetPolicy()
        self.cost_model = cost_model or TierCostModel()
        self._latencies: List[float] = []

    def get_max_feasible_tier(self, elapsed_ms: float) -> ExecutionTier:
        """
        Returns the highest tier whose nominal latency will not exceed the remaining SLA budget.
        """
        remaining_budget = max(0.0, self.policy.max_sla_deadline_ms - elapsed_ms)

        for tier in [
            ExecutionTier.TIER_4_CAUSAL_FORENSIC,
            ExecutionTier.TIER_3_DEEP_GNN,
            ExecutionTier.TIER_2_SHALLOW_ML,
            ExecutionTier.TIER_1_STREAMING_SKETCH,
            ExecutionTier.TIER_0_STATELESS,
        ]:
            spec = self.cost_model.get_spec(tier)
            if spec.nominal_latency_ms <= remaining_budget:
                return tier

        return ExecutionTier.TIER_0_STATELESS

    def record_event_latency(self, latency_ms: float) -> None:
        """Records an observed event processing latency."""
        self._latencies.append(latency_ms)

    def compute_distribution_metrics(self) -> LatencyDistributionMetrics:
        """Calculates p50, p90, p95, p99 and SLA violations across recorded latencies."""
        if not self._latencies:
            return LatencyDistributionMetrics(
                total_events=0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p90_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                max_latency_ms=0.0,
                sla_violations_count=0,
                sla_violation_rate_pct=0.0,
            )

        arr = np.array(self._latencies, dtype=np.float64)
        violations = int(np.sum(arr > self.policy.target_p99_sla_ms))
        v_rate = round((violations / len(arr)) * 100.0, 2)

        return LatencyDistributionMetrics(
            total_events=len(arr),
            mean_latency_ms=round(float(np.mean(arr)), 3),
            p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
            p90_latency_ms=round(float(np.percentile(arr, 90)), 3),
            p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
            p99_latency_ms=round(float(np.percentile(arr, 99)), 3),
            max_latency_ms=round(float(np.max(arr)), 3),
            sla_violations_count=violations,
            sla_violation_rate_pct=v_rate,
        )
