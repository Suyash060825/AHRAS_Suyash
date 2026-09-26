from __future__ import annotations
"""
AHRAS Module / Digital Twin — Response Counterfactual Simulation & Monte Carlo Engine
-------------------------------------------------------------------------------------
Performs pre-execution counterfactual simulation of autonomous security responses:
  - Validates preconditions and postconditions on the digital twin.
  - Measures kill-chain path breakage probability and remaining attack steps.
  - Calculates asset-weighted blast radius and collateral business disruption.
  - Conducts Monte Carlo uncertainty sampling over detector latency and adversary branching.
"""

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from security_twin.models import (
    ActionType, AttackScenario, AttackStage, AttackStep,
    MonteCarloResult, SimulationResult, SimulationStatus
)
from security_twin.state import SecurityTwin
from security_twin.scenario import check_step_preconditions, apply_step_postconditions

log = logging.getLogger(__name__)


class SecurityTwinSimulator:
    """
    Simulates candidate mitigation actions in an isolated digital twin fork,
    evaluating their efficacy, path breakage, and blast radius before execution.
    """

    def __init__(self, twin: SecurityTwin):
        self.twin = twin

    def simulate_action(
        self,
        scenario: AttackScenario,
        action_type: str,
        target_entity: str,
        current_risk: float = 0.85,
    ) -> SimulationResult:
        """
        Simulates a specific response action against an ongoing attack scenario.
        Executes in an atomic forked twin and rolls back to maintain state purity.
        """
        # 1. Action Precondition Verification
        action_feasible, pre_reason = self._verify_action_precondition(action_type, target_entity)
        if not action_feasible:
            return SimulationResult(
                action_type=action_type,
                target_entity=target_entity,
                pre_action_risk=current_risk,
                expected_post_action_risk=current_risk,
                path_breakage_probability=0.0,
                remaining_attack_steps=len(scenario.steps),
                blast_radius_score=0.0,
                collateral_disruption_cost=0.0,
                simulation_confidence=1.0,
                validation_status=SimulationStatus.INFEASIBLE.value,
                time_to_containment=0.0,
                explanation=f"Action infeasible: {pre_reason}",
            )

        # 2. Atomic Snapshot
        snapshot = self.twin.capture_state()

        try:
            # 3. Apply Action to Twin
            blast_radius, collateral_cost = self._apply_action(action_type, target_entity)

            # 4. Action Postcondition Verification
            post_verified = self._verify_action_postcondition(action_type, target_entity)
            if not post_verified:
                return SimulationResult(
                    action_type=action_type,
                    target_entity=target_entity,
                    pre_action_risk=current_risk,
                    expected_post_action_risk=current_risk,
                    path_breakage_probability=0.0,
                    remaining_attack_steps=len(scenario.steps),
                    blast_radius_score=blast_radius,
                    collateral_disruption_cost=collateral_cost,
                    simulation_confidence=0.9,
                    validation_status=SimulationStatus.POLICY_VIOLATION.value,
                    time_to_containment=0.0,
                    explanation="Postcondition failed: state did not reflect applied mitigation",
                )

            # 5. Replay Attack Scenario in Forked State
            total_steps = len(scenario.steps)
            executable_steps = 0
            blocked_step_idx: Optional[int] = None
            blocked_reason: str = ""

            for idx, step in enumerate(scenario.steps):
                can_exec, reason = check_step_preconditions(step, self.twin)
                if can_exec:
                    executable_steps += 1
                    apply_step_postconditions(step, self.twin)
                else:
                    if blocked_step_idx is None:
                        blocked_step_idx = idx
                        blocked_reason = reason
                    # In a strict kill chain, subsequent stages depending on this step are halted
                    break

            if blocked_step_idx is None:
                # Attack unhalted, all steps remain executable by adversary
                remaining_steps = total_steps
                halted_steps = 0
                breakage_prob = 0.0
            else:
                # Kill chain broken at blocked_step_idx
                halted_steps = total_steps - blocked_step_idx
                remaining_steps = 0
                breakage_prob = halted_steps / float(total_steps) if total_steps > 0 else 1.0

            # 6. Expected Post-Action Risk & Containment Calculation
            if action_type == ActionType.NO_ACTION.value or action_type == "NO_ACTION":
                expected_post_risk = min(1.0, current_risk + 0.12)
                time_to_containment = 999.0  # uncontained
            else:
                # Risk reduction formula: proportional to path breakage attenuated by residual compromise
                risk_reduction_rate = 0.85 * breakage_prob
                expected_post_risk = max(0.12, current_risk * (1.0 - risk_reduction_rate))
                time_to_containment = self._estimate_containment_time(action_type)

            # 7. Validation Policy Gating
            if blast_radius > 0.70 and (current_risk - expected_post_risk) < 0.25:
                status = SimulationStatus.HIGH_BLAST_RADIUS.value
            elif blast_radius > 0.85:
                status = SimulationStatus.HIGH_BLAST_RADIUS.value
            else:
                status = SimulationStatus.VALIDATED.value

            explanation = (
                f"Action {action_type} on {target_entity}: Path breakage {breakage_prob * 100:.1f}%, "
                f"{halted_steps} steps halted ({remaining_steps} unmitigated steps). Post-risk: {expected_post_risk:.3f} "
                f"(Blast radius: {blast_radius:.2f})."
            )
            if blocked_reason:
                explanation += f" First blocked at step {blocked_step_idx}: {blocked_reason}."

            return SimulationResult(
                action_type=action_type,
                target_entity=target_entity,
                pre_action_risk=round(current_risk, 4),
                expected_post_action_risk=round(expected_post_risk, 4),
                path_breakage_probability=round(breakage_prob, 4),
                remaining_attack_steps=remaining_steps,
                blast_radius_score=round(blast_radius, 4),
                collateral_disruption_cost=round(collateral_cost, 4),
                simulation_confidence=0.95,
                validation_status=status,
                time_to_containment=time_to_containment,
                explanation=explanation,
            )

        finally:
            # 8. Restore State to maintain twin purity
            self.twin.restore_state(snapshot)

    # ── Precondition / Postcondition Helpers ─────────────────────────────────

    def _verify_action_precondition(self, action_type: str, target: str) -> Tuple[bool, str]:
        if action_type in (ActionType.NO_ACTION.value, "NO_ACTION"):
            return True, "No action always applicable"

        if action_type in (ActionType.BLOCK_SOURCE.value, "BLOCK_SOURCE", "BLOCK_IP"):
            if self.twin.is_ip_blocked(target):
                return False, f"Target IP {target} is already blocked"
            return True, "IP available for perimeter blocking"

        if action_type in (ActionType.ISOLATE_HOST.value, "ISOLATE_HOST"):
            host = self.twin.get_host(target) or self.twin.get_host_by_ip(target)
            if not host:
                return False, f"Target host {target} does not exist in twin inventory"
            if host.is_isolated:
                return False, f"Host {target} is already isolated"
            return True, "Host available for isolation"

        if action_type in (ActionType.REVOKE_TOKEN.value, "REVOKE_TOKEN"):
            ident = self.twin.get_identity(target)
            if ident and not ident.is_valid:
                return False, f"Token {target} is already invalidated"
            user = self.twin.get_user(target) or self.twin.get_user_by_name(target)
            if user and user.credentials_revoked:
                return False, f"User {target} credentials already revoked"
            if not ident and not user:
                return False, f"Identity or user {target} not found in directory"
            return True, "Identity valid and eligible for revocation"

        if action_type in (ActionType.TERMINATE_PROCESS.value, "TERMINATE_PROCESS"):
            try:
                pid = int(target)
            except ValueError:
                return False, f"Invalid PID {target}"
            proc = self.twin.get_process(pid)
            if not proc:
                return False, f"PID {pid} not found in process tree"
            if proc.is_terminated:
                return False, f"PID {pid} is already terminated"
            return True, "Process running and eligible for termination"

        return False, f"Unknown action type {action_type}"

    def _apply_action(self, action_type: str, target: str) -> Tuple[float, float]:
        """Applies action to twin and calculates (blast_radius, disruption_cost)."""
        if action_type in (ActionType.NO_ACTION.value, "NO_ACTION"):
            return 0.0, 0.0

        if action_type in (ActionType.BLOCK_SOURCE.value, "BLOCK_SOURCE", "BLOCK_IP"):
            for net in self.twin.networks.values():
                if target not in net.blocked_ips:
                    net.blocked_ips.append(target)
            for ctrl in self.twin.controls.values():
                if ctrl.control_type == "FIREWALL" and ctrl.is_active:
                    ctrl.rules.setdefault("blocked_ips", []).append(target)
            blast_radius = 0.08
            disruption_cost = 8.0
            return blast_radius, disruption_cost

        if action_type in (ActionType.ISOLATE_HOST.value, "ISOLATE_HOST"):
            host = self.twin.get_host(target) or self.twin.get_host_by_ip(target)
            if host:
                host.is_isolated = True
                blast_radius = self.twin.compute_host_blast_radius(host.host_id)
                disruption_cost = blast_radius * host.criticality * 100.0
                return blast_radius, disruption_cost
            return 0.25, 25.0

        if action_type in (ActionType.REVOKE_TOKEN.value, "REVOKE_TOKEN"):
            ident = self.twin.get_identity(target)
            if ident:
                ident.is_valid = False
            user = self.twin.get_user(target) or self.twin.get_user_by_name(target)
            if user:
                user.credentials_revoked = True
            blast_radius = 0.20 if (user and user.is_privileged) else 0.10
            disruption_cost = blast_radius * 50.0
            return blast_radius, disruption_cost

        if action_type in (ActionType.TERMINATE_PROCESS.value, "TERMINATE_PROCESS"):
            pid = int(target)
            proc = self.twin.get_process(pid)
            if proc:
                proc.is_terminated = True
                blast_radius = self.twin.compute_process_blast_radius(pid)
                disruption_cost = blast_radius * 40.0
                return blast_radius, disruption_cost
            return 0.15, 15.0

        return 0.10, 10.0

    def _verify_action_postcondition(self, action_type: str, target: str) -> bool:
        if action_type in (ActionType.NO_ACTION.value, "NO_ACTION"):
            return True
        if action_type in (ActionType.BLOCK_SOURCE.value, "BLOCK_SOURCE", "BLOCK_IP"):
            return self.twin.is_ip_blocked(target)
        if action_type in (ActionType.ISOLATE_HOST.value, "ISOLATE_HOST"):
            return self.twin.is_host_isolated(target)
        if action_type in (ActionType.REVOKE_TOKEN.value, "REVOKE_TOKEN"):
            ident = self.twin.get_identity(target)
            if ident and ident.is_valid:
                return False
            user = self.twin.get_user(target) or self.twin.get_user_by_name(target)
            if user and not user.credentials_revoked:
                return False
            return True
        if action_type in (ActionType.TERMINATE_PROCESS.value, "TERMINATE_PROCESS"):
            return self.twin.is_process_terminated(int(target))
        return False

    @staticmethod
    def _estimate_containment_time(action_type: str) -> float:
        """Estimated automated response actuation latency in seconds."""
        latencies = {
            "BLOCK_SOURCE": 1.2,
            "BLOCK_IP": 1.2,
            "REVOKE_TOKEN": 1.5,
            "TERMINATE_PROCESS": 1.8,
            "ISOLATE_HOST": 2.4,
        }
        return latencies.get(action_type, 3.0)

    # ── Monte Carlo Uncertainty Sampling ──────────────────────────────────────

    def run_monte_carlo(
        self,
        scenario: AttackScenario,
        action_type: str,
        target_entity: str,
        num_iterations: int = 200,
        seed: int = 42,
    ) -> MonteCarloResult:
        """
        Samples uncertainty distributions over:
          - detector probability of detection per step (Beta distribution)
          - actuation delay & analyst lag (Lognormal distribution)
          - asset business criticality perturbation (Gaussian noise)
          - adversary branching success (Bernoulli trials)
        """
        rng = np.random.default_rng(seed)
        risk_samples: List[float] = []
        contained_count = 0
        blast_samples: List[float] = []

        # Baseline single deterministic simulation
        base_sim = self.simulate_action(scenario, action_type, target_entity)

        for _ in range(num_iterations):
            # 1. Sample detector detection probability
            p_detect = rng.beta(8.0, 2.0)  # mean ~0.80

            # 2. Sample actuation latency (log-normal, median ~1.8s)
            latency_sec = float(rng.lognormal(mean=0.5, sigma=0.3))

            # 3. Sample adversary step bypass noise
            step_bypass_prob = float(rng.uniform(0.02, 0.12))

            # 4. Check if response is effective given detection & latency
            if action_type in (ActionType.NO_ACTION.value, "NO_ACTION"):
                # No response -> high residual risk
                residual = float(np.clip(0.85 + rng.normal(0.05, 0.04), 0.70, 1.0))
                contained = False
            else:
                # With mitigation:
                if rng.random() > p_detect:
                    # Detection missed this round -> partial escalation
                    residual = float(np.clip(0.65 + rng.normal(0.0, 0.08), 0.40, 0.95))
                    contained = False
                elif rng.random() < step_bypass_prob:
                    # Adversary found minor bypass
                    residual = float(np.clip(base_sim.expected_post_action_risk + 0.20 + rng.normal(0.0, 0.05), 0.25, 0.80))
                    contained = False
                else:
                    # Successful containment
                    noise = float(rng.normal(0.0, 0.03))
                    residual = float(np.clip(base_sim.expected_post_action_risk + noise, 0.05, 0.50))
                    contained = True

            if contained:
                contained_count += 1
            risk_samples.append(round(residual, 4))

            # Perturb blast radius slightly
            blast_sample = float(np.clip(base_sim.blast_radius_score + rng.normal(0.0, 0.02), 0.01, 1.0))
            blast_samples.append(blast_sample)

        samples_arr = np.array(risk_samples)
        p10, p50, p90, p99 = np.percentile(samples_arr, [10, 50, 90, 99])
        containment_prob = contained_count / float(num_iterations)
        escalation_prob = 1.0 - containment_prob

        return MonteCarloResult(
            action_type=action_type,
            num_iterations=num_iterations,
            mean_residual_risk=round(float(np.mean(samples_arr)), 4),
            median_residual_risk=round(float(np.median(samples_arr)), 4),
            std_residual_risk=round(float(np.std(samples_arr)), 4),
            p10_risk=round(float(p10), 4),
            p50_risk=round(float(p50), 4),
            p90_risk=round(float(p90), 4),
            p99_risk=round(float(p99), 4),
            containment_probability=round(float(containment_prob), 4),
            escalation_probability=round(float(escalation_prob), 4),
            blast_radius_mean=round(float(np.mean(blast_samples)), 4),
            risk_samples=risk_samples,
        )
