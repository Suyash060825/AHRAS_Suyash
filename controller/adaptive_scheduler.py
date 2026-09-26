"""
AHRAS Adaptive Scheduler
------------------------
Monitors real-time system resource pressure (CPU utilization, queue depth, RAM)
and computes dynamic load-shedding penalties to prevent latency collapse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from controller.cost_model import ExecutionTier


@dataclass
class SystemLoadState:
    """
    Real-time operating telemetry of the inspection host.
    """
    cpu_utilization_pct: float = 25.0      # 0 to 100%
    queue_depth: int = 10                  # Current ingress queue backlog
    queue_capacity: int = 1000             # Max queue buffer depth
    memory_used_mb: float = 120.0          # Current process heap RAM
    memory_limit_mb: float = 1024.0        # System cgroup RAM cap
    current_eps: float = 2500.0            # Ingestion rate (events/sec)

    @property
    def is_congested(self) -> bool:
        return self.cpu_utilization_pct > 75.0 or (self.queue_depth / max(1, self.queue_capacity)) > 0.60

    @property
    def is_critical_overload(self) -> bool:
        return self.cpu_utilization_pct > 90.0 or (self.queue_depth / max(1, self.queue_capacity)) > 0.85


@dataclass
class ScheduledTaskVerdict:
    """
    Scheduler recommendation for an incoming event.
    """
    max_permitted_tier: ExecutionTier
    load_shedding_active: bool
    effective_cpu_penalty: float
    effective_queue_penalty: float


class AdaptiveScheduler:
    """
    Dynamically adjusts available execution tiers based on closed-loop feedback.
    """
    def __init__(
        self,
        base_queue_weight: float = 0.50,
        base_cpu_weight: float = 0.50,
    ) -> None:
        self.base_queue_weight = base_queue_weight
        self.base_cpu_weight = base_cpu_weight

    def evaluate_load(self, state: SystemLoadState) -> ScheduledTaskVerdict:
        """
        Calculates dynamic penalty weights and assigns maximum allowed tier under current load.
        """
        q_ratio = min(1.0, state.queue_depth / max(1, state.queue_capacity))
        cpu_ratio = min(0.99, state.cpu_utilization_pct / 100.0)

        # Exponential queue penalty: lambda_1 = base * exp(q_ratio)
        lambda_q = self.base_queue_weight * math.exp(q_ratio)

        # Non-linear CPU barrier penalty: lambda_2 = base / (1 - cpu_ratio)
        lambda_cpu = self.base_cpu_weight / (1.0 - cpu_ratio)

        # Assign tier limits based on load state
        if state.is_critical_overload:
            # Under severe overload, strictly cap inline analysis at Tier 1 (Fast Path)
            max_tier = ExecutionTier.TIER_1_STREAMING_SKETCH
            load_shedding = True
        elif state.is_congested:
            # Under moderate congestion, cap at Tier 2 (Shallow ML)
            max_tier = ExecutionTier.TIER_2_SHALLOW_ML
            load_shedding = True
        else:
            # Normal operating conditions: full Tier 4 depth accessible
            max_tier = ExecutionTier.TIER_4_CAUSAL_FORENSIC
            load_shedding = False

        return ScheduledTaskVerdict(
            max_permitted_tier=max_tier,
            load_shedding_active=load_shedding,
            effective_cpu_penalty=round(lambda_cpu, 4),
            effective_queue_penalty=round(lambda_q, 4),
        )
