"""
AHRAS Resource-Aware Multi-Tier Controller
------------------------------------------
Dynamically arbitrates computational analysis depth across 5 execution tiers
(Tier 0 to Tier 4) based on real-time CPU load, queue latency budgets,
event threat priors, and conformal confidence guarantees.

Formal Utility Optimization:
    d* = argmax_{d <= d_max} [ SecurityValue(e, d)
                             - lambda_1(L) * C_lat(d)
                             - lambda_2(C) * C_cpu(d)
                             - lambda_3(M) * C_mem(d) ]
"""

from __future__ import annotations

import time
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from controller.cost_model import ExecutionTier, TierSpecification, TierCostModel, DEFAULT_TIER_SPECS
from controller.latency_budget import LatencyBudgetManager, LatencyBudgetPolicy
from controller.adaptive_scheduler import AdaptiveScheduler, SystemLoadState, ScheduledTaskVerdict


@dataclass
class ControllerRoutingDecision:
    """
    Result of multi-tier optimization decision for an incoming event.
    """
    event_id: str
    selected_tier: ExecutionTier
    max_permitted_tier: ExecutionTier
    budget_tier_limit: ExecutionTier
    utility_score: float
    estimated_latency_ms: float
    estimated_cpu_cost: float
    exit_reason: str
    confidence: float
    risk_prior: float
    detection_capability: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "selected_tier": self.selected_tier.value,
            "tier_name": self.selected_tier.name,
            "max_permitted_tier": self.max_permitted_tier.value,
            "budget_tier_limit": self.budget_tier_limit.value,
            "utility_score": round(self.utility_score, 4),
            "estimated_latency_ms": round(self.estimated_latency_ms, 3),
            "estimated_cpu_cost": round(self.estimated_cpu_cost, 3),
            "exit_reason": self.exit_reason,
            "confidence": round(self.confidence, 4),
            "risk_prior": round(self.risk_prior, 4),
            "detection_capability": round(self.detection_capability, 4),
            "details": self.details,
        }


class ResourceAwareTierController:
    """
    Online multi-tier arbitration controller.
    """
    def __init__(
        self,
        cost_model: Optional[TierCostModel] = None,
        latency_manager: Optional[LatencyBudgetManager] = None,
        scheduler: Optional[AdaptiveScheduler] = None,
        lambda_lat: float = 0.08,
        lambda_cpu: float = 0.05,
        lambda_mem: float = 0.002,
        early_exit_threshold: float = 0.92,
    ) -> None:
        self.cost_model = cost_model or TierCostModel()
        self.latency_manager = latency_manager or LatencyBudgetManager()
        self.scheduler = scheduler or AdaptiveScheduler()
        self.lambda_lat = lambda_lat
        self.lambda_cpu = lambda_cpu
        self.lambda_mem = lambda_mem
        self.early_exit_threshold = early_exit_threshold

    def compute_risk_prior(self, event: Dict[str, Any]) -> Tuple[float, float]:
        """
        Estimates a threat risk prior in [0, 1] and confidence score in [0, 1].
        Extracts structural, port, protocol, command line, or severity clues.
        """
        # Base prior from severity_id or explicit indicators
        sev = float(event.get("severity_id", event.get("severity", 1)))
        raw_risk = min(1.0, (sev - 1.0) / 4.0)

        # Check explicit attack indicators
        cmd = str(event.get("command", event.get("actor", {}).get("process", {}).get("cmd_line", ""))).lower()
        dst_port = int(event.get("dst_port", event.get("dst_endpoint", {}).get("port", 0) or 0))
        proto = str(event.get("protocol", "")).upper()

        high_risk_terms = ["mimikatz", "vssadmin", "powershell -enc", "whoami /all", "certutil -urlcache", "nc -e", "/bin/sh -i"]
        if any(term in cmd for term in high_risk_terms):
            raw_risk = max(raw_risk, 0.95)

        suspicious_ports = [4444, 1337, 6667, 31337, 8888]
        if dst_port in suspicious_ports:
            raw_risk = max(raw_risk, 0.85)

        # Confidence: High if clear benign (routine web/dns) or clear signature attack
        if raw_risk >= 0.90 or raw_risk <= 0.05:
            confidence = 0.95
        elif raw_risk >= 0.60:
            confidence = 0.70
        else:
            confidence = 0.50

        # Optional explicit confidence in event
        if "confidence" in event:
            confidence = float(event["confidence"])
        if "risk_prior" in event:
            raw_risk = float(event["risk_prior"])

        return raw_risk, confidence

    def compute_tier_utility(
        self,
        tier: ExecutionTier,
        risk_prior: float,
        confidence: float,
        lambda_q: float,
        lambda_cpu: float,
    ) -> float:
        """
        Calculates net security utility of assigning an event to candidate tier.
        U(d) = SecurityValue(e, d) - CostPenalties(d)
        """
        spec = self.cost_model.get_spec(tier)

        # Security value increases with detection capability, especially when threat prior is elevated
        # If risk is near 0 and confidence is high, lower tiers suffice (high capability not needed)
        security_value = (risk_prior * spec.detection_capability) + (1.0 - risk_prior) * 0.40

        # Cost penalties scaled by dynamic scheduler weights
        lat_cost = (spec.nominal_latency_ms / 10.0) * (self.lambda_lat * lambda_q)
        cpu_cost = (spec.cpu_cost_factor / 1.0) * (self.lambda_cpu * lambda_cpu)
        mem_cost = (spec.memory_mb / 100.0) * self.lambda_mem

        total_penalty = lat_cost + cpu_cost + mem_cost
        return security_value - total_penalty

    def arbitrate_tier(
        self,
        event: Dict[str, Any],
        system_state: Optional[SystemLoadState] = None,
        elapsed_ms: float = 0.0,
    ) -> ControllerRoutingDecision:
        """
        Arbitrates execution tier for an event under current load and SLA constraints.
        """
        event_id = str(event.get("event_id", uuid.uuid4()))
        risk_prior, confidence = self.compute_risk_prior(event)

        state = system_state or SystemLoadState()
        sched_verdict = self.scheduler.evaluate_load(state)
        budget_limit = self.latency_manager.get_max_feasible_tier(elapsed_ms)

        # Highest permitted tier is minimum of load limit and latency SLA limit
        ceiling_tier_val = min(sched_verdict.max_permitted_tier.value, budget_limit.value)
        ceiling_tier = ExecutionTier(ceiling_tier_val)

        # 1. Early-Exit Shortcut Check:
        # If clear benign and low risk, Tier 0 (stateless) or Tier 1 (sketch) can safely resolve
        if confidence >= self.early_exit_threshold and risk_prior <= 0.08:
            selected_tier = ExecutionTier.TIER_0_STATELESS
            spec = self.cost_model.get_spec(selected_tier)
            return ControllerRoutingDecision(
                event_id=event_id,
                selected_tier=selected_tier,
                max_permitted_tier=sched_verdict.max_permitted_tier,
                budget_tier_limit=budget_limit,
                utility_score=0.95,
                estimated_latency_ms=spec.nominal_latency_ms,
                estimated_cpu_cost=spec.cpu_cost_factor,
                exit_reason="early_exit_confident_benign",
                confidence=confidence,
                risk_prior=risk_prior,
                detection_capability=spec.detection_capability,
                details={"sched_verdict": sched_verdict.__dict__},
            )

        # If clear known signature match (e.g. known malicious IOC), Tier 0 / Tier 1 can decisively alert
        if risk_prior >= 0.95 and confidence >= self.early_exit_threshold:
            selected_tier = ExecutionTier.TIER_0_STATELESS
            spec = self.cost_model.get_spec(selected_tier)
            return ControllerRoutingDecision(
                event_id=event_id,
                selected_tier=selected_tier,
                max_permitted_tier=sched_verdict.max_permitted_tier,
                budget_tier_limit=budget_limit,
                utility_score=0.98,
                estimated_latency_ms=spec.nominal_latency_ms,
                estimated_cpu_cost=spec.cpu_cost_factor,
                exit_reason="early_exit_decisive_signature",
                confidence=confidence,
                risk_prior=risk_prior,
                detection_capability=spec.detection_capability,
                details={"sched_verdict": sched_verdict.__dict__},
            )

        # 2. Utility Optimization across accessible tiers [0 .. ceiling_tier]
        candidate_tiers = [ExecutionTier(t) for t in range(ceiling_tier_val + 1)]
        best_tier = ExecutionTier.TIER_0_STATELESS
        best_utility = -float("inf")

        for tier in candidate_tiers:
            u = self.compute_tier_utility(
                tier=tier,
                risk_prior=risk_prior,
                confidence=confidence,
                lambda_q=sched_verdict.effective_queue_penalty,
                lambda_cpu=sched_verdict.effective_cpu_penalty,
            )
            if u > best_utility:
                best_utility = u
                best_tier = tier

        # Check if capped by load shedding or SLA budget
        exit_reason = "utility_optimal"
        if ceiling_tier_val < ExecutionTier.TIER_4_CAUSAL_FORENSIC.value:
            if sched_verdict.load_shedding_active and best_tier == ceiling_tier:
                exit_reason = "load_shedding_constrained"
            elif budget_limit.value < ExecutionTier.TIER_4_CAUSAL_FORENSIC.value and best_tier == ceiling_tier:
                exit_reason = "latency_budget_constrained"

        chosen_spec = self.cost_model.get_spec(best_tier)
        return ControllerRoutingDecision(
            event_id=event_id,
            selected_tier=best_tier,
            max_permitted_tier=sched_verdict.max_permitted_tier,
            budget_tier_limit=budget_limit,
            utility_score=best_utility,
            estimated_latency_ms=chosen_spec.nominal_latency_ms,
            estimated_cpu_cost=chosen_spec.cpu_cost_factor,
            exit_reason=exit_reason,
            confidence=confidence,
            risk_prior=risk_prior,
            detection_capability=chosen_spec.detection_capability,
            details={
                "candidate_tiers": [t.value for t in candidate_tiers],
                "effective_cpu_penalty": sched_verdict.effective_cpu_penalty,
                "effective_queue_penalty": sched_verdict.effective_queue_penalty,
            },
        )

    def process_event(
        self,
        event: Dict[str, Any],
        system_state: Optional[SystemLoadState] = None,
    ) -> Tuple[ControllerRoutingDecision, float]:
        """
        Executes routing decision and records simulated execution latency in budget manager.
        Returns: (routing_decision, observed_latency_ms)
        """
        t0 = time.perf_counter()
        decision = self.arbitrate_tier(event, system_state)
        # Record nominal latency plus microsecond routing overhead
        routing_overhead_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms = decision.estimated_latency_ms + routing_overhead_ms
        self.latency_manager.record_event_latency(total_latency_ms)
        return decision, total_latency_ms
