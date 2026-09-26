"""
Tests for AHRAS Resource-Aware Multi-Tier Controller (Research Frontier E / EXP-26)
-----------------------------------------------------------------------------------
Validates:
  1. TierCostModel specification integrity and batch cost estimates.
  2. LatencyBudgetManager SLA filtering and quantile distribution computations.
  3. AdaptiveScheduler dynamic penalty computation and congestion load-shedding.
  4. ResourceAwareTierController utility optimization and early-exit routing.
  5. Load-shedding under critical CPU pressure and latency budget exhaustion.
"""

import pytest
import numpy as np
from controller.cost_model import (
    ExecutionTier,
    TierSpecification,
    TierCostModel,
    DEFAULT_TIER_SPECS,
)
from controller.latency_budget import (
    LatencyBudgetManager,
    LatencyBudgetPolicy,
    LatencyDistributionMetrics,
)
from controller.adaptive_scheduler import (
    AdaptiveScheduler,
    SystemLoadState,
    ScheduledTaskVerdict,
)
from controller.tier_controller import (
    ResourceAwareTierController,
    ControllerRoutingDecision,
)


class TestTierCostModel:
    def test_default_tier_specs_exist_and_ordered(self):
        model = TierCostModel()
        for tier in ExecutionTier:
            spec = model.get_spec(tier)
            assert spec.tier == tier
            assert spec.nominal_latency_ms >= 0.0
            assert spec.cpu_cost_factor > 0.0
            assert spec.memory_mb > 0.0

        # Latency must strictly monotonically increase with tier depth
        latencies = [model.get_spec(t).nominal_latency_ms for t in ExecutionTier]
        assert latencies == sorted(latencies)

        # Capabilities must strictly monotonically increase
        caps = [model.get_spec(t).detection_capability for t in ExecutionTier]
        assert caps == sorted(caps)

    def test_batch_cost_estimation(self):
        model = TierCostModel()
        cost_t0 = model.estimate_batch_cost(ExecutionTier.TIER_0_STATELESS, 100)
        cost_t3 = model.estimate_batch_cost(ExecutionTier.TIER_3_DEEP_GNN, 100)

        assert cost_t0["total_latency_ms"] < cost_t3["total_latency_ms"]
        assert cost_t0["cpu_work_units"] < cost_t3["cpu_work_units"]


class TestLatencyBudgetManager:
    def test_max_feasible_tier_under_budget(self):
        policy = LatencyBudgetPolicy(max_sla_deadline_ms=20.0)
        mgr = LatencyBudgetManager(policy=policy)

        # With 0ms elapsed, 20ms remaining: Tier 3 (18.5ms) is feasible, Tier 4 (85ms) is not
        feasible = mgr.get_max_feasible_tier(elapsed_ms=0.0)
        assert feasible == ExecutionTier.TIER_3_DEEP_GNN

        # With 18ms elapsed, only 2ms remaining: Tier 1 (0.35ms) is feasible, Tier 2 (2.8ms) is not
        feasible_low = mgr.get_max_feasible_tier(elapsed_ms=18.0)
        assert feasible_low == ExecutionTier.TIER_1_STREAMING_SKETCH

        # With 25ms elapsed (over budget): Tier 0 is fallback
        feasible_exhausted = mgr.get_max_feasible_tier(elapsed_ms=25.0)
        assert feasible_exhausted == ExecutionTier.TIER_0_STATELESS

    def test_distribution_metrics_and_sla_violations(self):
        policy = LatencyBudgetPolicy(target_p99_sla_ms=5.0)
        mgr = LatencyBudgetManager(policy=policy)

        # Add 95 events under 5ms, 5 events over 5ms
        for _ in range(95):
            mgr.record_event_latency(1.5)
        for _ in range(5):
            mgr.record_event_latency(8.0)

        metrics = mgr.compute_distribution_metrics()
        assert metrics.total_events == 100
        assert metrics.sla_violations_count == 5
        assert metrics.sla_violation_rate_pct == 5.0
        assert metrics.p50_latency_ms == pytest.approx(1.5, 0.1)
        assert metrics.p99_latency_ms > 5.0


class TestAdaptiveScheduler:
    def test_normal_load_state(self):
        scheduler = AdaptiveScheduler()
        normal_state = SystemLoadState(cpu_utilization_pct=30.0, queue_depth=20, queue_capacity=1000)
        verdict = scheduler.evaluate_load(normal_state)

        assert not verdict.load_shedding_active
        assert verdict.max_permitted_tier == ExecutionTier.TIER_4_CAUSAL_FORENSIC
        assert verdict.effective_cpu_penalty < 1.0

    def test_congested_load_state(self):
        scheduler = AdaptiveScheduler()
        congested_state = SystemLoadState(cpu_utilization_pct=80.0, queue_depth=700, queue_capacity=1000)
        verdict = scheduler.evaluate_load(congested_state)

        assert verdict.load_shedding_active
        assert verdict.max_permitted_tier == ExecutionTier.TIER_2_SHALLOW_ML
        assert verdict.effective_cpu_penalty > 2.0

    def test_critical_overload_state(self):
        scheduler = AdaptiveScheduler()
        overload_state = SystemLoadState(cpu_utilization_pct=95.0, queue_depth=900, queue_capacity=1000)
        verdict = scheduler.evaluate_load(overload_state)

        assert verdict.load_shedding_active
        assert verdict.max_permitted_tier == ExecutionTier.TIER_1_STREAMING_SKETCH
        assert verdict.effective_cpu_penalty >= 5.0


class TestResourceAwareTierController:
    def test_early_exit_benign_event(self):
        controller = ResourceAwareTierController()
        benign_event = {
            "event_id": "test-benign-1",
            "command": "echo normal",
            "dst_port": 80,
            "severity_id": 1,
            "confidence": 0.98,
            "risk_prior": 0.01,
        }
        decision = controller.arbitrate_tier(benign_event)
        assert decision.selected_tier == ExecutionTier.TIER_0_STATELESS
        assert "early_exit" in decision.exit_reason

    def test_early_exit_decisive_attack(self):
        controller = ResourceAwareTierController()
        mimikatz_event = {
            "event_id": "test-mimi-1",
            "command": "powershell.exe mimikatz sekurlsa::logonpasswords",
            "severity_id": 5,
        }
        decision = controller.arbitrate_tier(mimikatz_event)
        assert decision.selected_tier == ExecutionTier.TIER_0_STATELESS
        assert "early_exit_decisive_signature" in decision.exit_reason

    def test_escalation_under_normal_load(self):
        controller = ResourceAwareTierController()
        # High risk, ambiguous confidence event under normal load -> escalates to Tier 3 or 4
        complex_event = {
            "event_id": "test-complex-1",
            "dst_port": 4444,
            "severity_id": 3,
            "risk_prior": 0.75,
            "confidence": 0.60,
        }
        state = SystemLoadState(cpu_utilization_pct=20.0, queue_depth=5)
        decision = controller.arbitrate_tier(complex_event, system_state=state)
        # Utility should favor higher tiers for high risk when resources are abundant
        assert decision.selected_tier.value >= ExecutionTier.TIER_2_SHALLOW_ML.value

    def test_load_shedding_caps_execution(self):
        controller = ResourceAwareTierController()
        complex_event = {
            "event_id": "test-complex-2",
            "dst_port": 4444,
            "severity_id": 4,
            "risk_prior": 0.85,
            "confidence": 0.55,
        }
        # Under critical overload (CPU 96%)
        overload_state = SystemLoadState(cpu_utilization_pct=96.0, queue_depth=950)
        decision = controller.arbitrate_tier(complex_event, system_state=overload_state)
        # Must be strictly capped at Tier 1 or 0
        assert decision.selected_tier.value <= ExecutionTier.TIER_1_STREAMING_SKETCH.value

    def test_process_event_end_to_end(self):
        controller = ResourceAwareTierController()
        event = {"event_id": "evt-123", "command": "python test.py", "severity_id": 1}
        decision, lat = controller.process_event(event)

        assert isinstance(decision, ControllerRoutingDecision)
        assert lat > 0.0
        metrics = controller.latency_manager.compute_distribution_metrics()
        assert metrics.total_events == 1
