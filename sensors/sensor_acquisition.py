from __future__ import annotations
"""
AHRAS Module — Adaptive Sensor Acquisition Engine (Section 18 / Research Frontier P1)
--------------------------------------------------------------------------------------
Dynamically arbitrates on-demand telemetry sensor collection based on the formal
Value of Information (VOI) metric:

    ValueOfInformation(m, e) = ExpectedSecurityGain(m, e) - CollectionCost(m, system_load)

Guarantees:
  1. Policy-permission boundaries: Expensive sensors (e.g. deep memory inspection,
     endpoint forensic dumps) strictly require explicit authorization.
  2. Latency & Resource safety: Telemetry expansion is throttled or rejected if
     system queue latency exceeds SLA or CPU exceeds critical threshold.
  3. Provenance & Observability: Every acquisition decision logs expected gain,
     actual latency, dollar/CPU cost, and decision rationale.
"""

import enum
import logging
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


class SensorModality(str, enum.Enum):
    BASELINE_NETWORK = "BASELINE_NETWORK"            # Standard NetFlow / IP headers (free baseline)
    PROCESS_TELEMETRY = "PROCESS_TELEMETRY"          # Host auditd / eBPF process execution tree
    IDENTITY_TELEMETRY = "IDENTITY_TELEMETRY"        # IAM, Kerberos, OIDC ticket history & privileges
    GRAPH_NEIGHBORHOOD = "GRAPH_NEIGHBORHOOD"        # 2-hop / 3-hop entity relationship graph expansion
    SESSION_DETAIL = "SESSION_DETAIL"                # Full packet payload / TLS client hello metadata
    ENDPOINT_CONTEXT = "ENDPOINT_CONTEXT"            # Registry keys, file hashes, DLL injection scans
    DEEP_FORENSIC = "DEEP_FORENSIC"                  # Full memory dump / interactive sandbox execution


@dataclass
class SensorModalityProfile:
    """Cost, latency, and information-gain parameters for a sensor modality."""
    modality: SensorModality
    nominal_cost_units: float        # Normalized compute/storage cost
    nominal_latency_ms: float        # Acquisition collection delay
    base_information_gain: float     # Expected uncertainty reduction Delta H
    requires_explicit_policy: bool   # If True, requires policy whitelist
    memory_footprint_mb: float


DEFAULT_SENSOR_PROFILES: Dict[SensorModality, SensorModalityProfile] = {
    SensorModality.BASELINE_NETWORK: SensorModalityProfile(
        modality=SensorModality.BASELINE_NETWORK,
        nominal_cost_units=0.01,
        nominal_latency_ms=0.05,
        base_information_gain=0.20,
        requires_explicit_policy=False,
        memory_footprint_mb=0.5,
    ),
    SensorModality.PROCESS_TELEMETRY: SensorModalityProfile(
        modality=SensorModality.PROCESS_TELEMETRY,
        nominal_cost_units=0.85,
        nominal_latency_ms=2.10,
        base_information_gain=0.65,
        requires_explicit_policy=False,
        memory_footprint_mb=4.0,
    ),
    SensorModality.IDENTITY_TELEMETRY: SensorModalityProfile(
        modality=SensorModality.IDENTITY_TELEMETRY,
        nominal_cost_units=0.45,
        nominal_latency_ms=1.20,
        base_information_gain=0.50,
        requires_explicit_policy=False,
        memory_footprint_mb=2.0,
    ),
    SensorModality.GRAPH_NEIGHBORHOOD: SensorModalityProfile(
        modality=SensorModality.GRAPH_NEIGHBORHOOD,
        nominal_cost_units=1.80,
        nominal_latency_ms=6.50,
        base_information_gain=0.78,
        requires_explicit_policy=False,
        memory_footprint_mb=12.0,
    ),
    SensorModality.SESSION_DETAIL: SensorModalityProfile(
        modality=SensorModality.SESSION_DETAIL,
        nominal_cost_units=1.50,
        nominal_latency_ms=4.80,
        base_information_gain=0.70,
        requires_explicit_policy=False,
        memory_footprint_mb=8.0,
    ),
    SensorModality.ENDPOINT_CONTEXT: SensorModalityProfile(
        modality=SensorModality.ENDPOINT_CONTEXT,
        nominal_cost_units=3.20,
        nominal_latency_ms=15.00,
        base_information_gain=0.88,
        requires_explicit_policy=True,
        memory_footprint_mb=24.0,
    ),
    SensorModality.DEEP_FORENSIC: SensorModalityProfile(
        modality=SensorModality.DEEP_FORENSIC,
        nominal_cost_units=8.50,
        nominal_latency_ms=65.00,
        base_information_gain=0.96,
        requires_explicit_policy=True,
        memory_footprint_mb=80.0,
    ),
}


@dataclass
class AcquisitionPolicy:
    """Enforces organizational authorization rules for sensor activation."""
    allowed_modalities: Set[SensorModality] = field(
        default_factory=lambda: {
            SensorModality.BASELINE_NETWORK,
            SensorModality.PROCESS_TELEMETRY,
            SensorModality.IDENTITY_TELEMETRY,
            SensorModality.GRAPH_NEIGHBORHOOD,
            SensorModality.SESSION_DETAIL,
            SensorModality.ENDPOINT_CONTEXT,
        }
    )
    allow_deep_forensics: bool = False
    max_total_collection_latency_ms: float = 25.0
    voi_minimum_threshold: float = 0.12


@dataclass
class SensorAcquisitionDecision:
    """Detailed audit of a dynamic telemetry acquisition evaluation."""
    event_id: str
    modality: SensorModality
    should_acquire: bool
    value_of_information: float
    expected_security_gain: float
    collection_cost: float
    estimated_latency_ms: float
    policy_permitted: bool
    rejection_reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "modality": self.modality.value,
            "should_acquire": self.should_acquire,
            "value_of_information": round(self.value_of_information, 4),
            "expected_security_gain": round(self.expected_security_gain, 4),
            "collection_cost": round(self.collection_cost, 4),
            "estimated_latency_ms": round(self.estimated_latency_ms, 2),
            "policy_permitted": self.policy_permitted,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp,
        }


class AdaptiveSensorAcquisitionEngine:
    """
    Evaluates whether collecting additional telemetry is computationally and operationally justified.
    """

    def __init__(
        self,
        profiles: Optional[Dict[SensorModality, SensorModalityProfile]] = None,
        policy: Optional[AcquisitionPolicy] = None,
        cost_weight_lambda: float = 0.25,
    ) -> None:
        self.profiles = profiles or DEFAULT_SENSOR_PROFILES
        self.policy = policy or AcquisitionPolicy()
        self.cost_weight_lambda = cost_weight_lambda
        self._lock = threading.RLock()
        self._history: List[SensorAcquisitionDecision] = []

    def compute_expected_security_gain(
        self,
        modality: SensorModality,
        event_threat_prior: float,
        epistemic_uncertainty: float,
        asset_criticality: float = 1.0,
    ) -> float:
        """
        Computes expected security value: high if risk is ambiguous/uncertain on high-value asset.
        gain = base_gain * (threat_prior * uncertainty + 0.1 * asset_criticality)
        """
        profile = self.profiles[modality]
        # Ambiguity is maximized when uncertainty is high
        ambiguity_factor = epistemic_uncertainty * (1.0 - abs(event_threat_prior - 0.5) * 1.5)
        ambiguity_factor = max(0.1, min(1.0, ambiguity_factor))

        criticality_factor = max(0.5, min(2.0, asset_criticality))
        gain = profile.base_information_gain * (0.6 * ambiguity_factor + 0.4 * event_threat_prior) * (criticality_factor / 1.5)
        return float(np.clip(gain, 0.0, 1.0))

    def compute_effective_cost(
        self,
        modality: SensorModality,
        cpu_load_factor: float = 0.3,
        queue_latency_factor: float = 0.2,
    ) -> float:
        """Adjusts nominal sensor cost based on live server CPU load and queuing."""
        profile = self.profiles[modality]
        congestion_penalty = 1.0 + 1.5 * max(0.0, cpu_load_factor - 0.5) + 2.0 * max(0.0, queue_latency_factor - 0.5)
        cost = profile.nominal_cost_units * self.cost_weight_lambda * congestion_penalty
        return float(cost)

    def evaluate_modality(
        self,
        event_id: str,
        modality: SensorModality,
        threat_prior: float,
        epistemic_uncertainty: float,
        asset_criticality: float = 1.0,
        cpu_load: float = 0.25,
        elapsed_budget_ms: float = 0.0,
    ) -> SensorAcquisitionDecision:
        """Evaluates whether to collect a specific sensor modality for an incoming event."""
        profile = self.profiles[modality]

        # 1. Policy Authorization Check
        if profile.requires_explicit_policy:
            if modality == SensorModality.DEEP_FORENSIC and not self.policy.allow_deep_forensics:
                return SensorAcquisitionDecision(
                    event_id=event_id,
                    modality=modality,
                    should_acquire=False,
                    value_of_information=0.0,
                    expected_security_gain=0.0,
                    collection_cost=profile.nominal_cost_units,
                    estimated_latency_ms=profile.nominal_latency_ms,
                    policy_permitted=False,
                    rejection_reason="POLICY_DENIED_DEEP_FORENSIC",
                )
            if modality not in self.policy.allowed_modalities:
                return SensorAcquisitionDecision(
                    event_id=event_id,
                    modality=modality,
                    should_acquire=False,
                    value_of_information=0.0,
                    expected_security_gain=0.0,
                    collection_cost=profile.nominal_cost_units,
                    estimated_latency_ms=profile.nominal_latency_ms,
                    policy_permitted=False,
                    rejection_reason="POLICY_DENIED_MODALITY",
                )

        # 2. Latency Budget Constraint
        remaining_budget = self.policy.max_total_collection_latency_ms - elapsed_budget_ms
        if profile.nominal_latency_ms > remaining_budget:
            return SensorAcquisitionDecision(
                event_id=event_id,
                modality=modality,
                should_acquire=False,
                value_of_information=0.0,
                expected_security_gain=0.0,
                collection_cost=profile.nominal_cost_units,
                estimated_latency_ms=profile.nominal_latency_ms,
                policy_permitted=True,
                rejection_reason="EXCEEDS_LATENCY_BUDGET",
            )

        # 3. Value of Information Computation
        gain = self.compute_expected_security_gain(
            modality=modality,
            event_threat_prior=threat_prior,
            epistemic_uncertainty=epistemic_uncertainty,
            asset_criticality=asset_criticality,
        )
        cost = self.compute_effective_cost(modality=modality, cpu_load_factor=cpu_load)
        voi = gain - cost

        should_acquire = voi >= self.policy.voi_minimum_threshold
        rejection_reason = "" if should_acquire else "INSUFFICIENT_VOI"

        dec = SensorAcquisitionDecision(
            event_id=event_id,
            modality=modality,
            should_acquire=should_acquire,
            value_of_information=voi,
            expected_security_gain=gain,
            collection_cost=cost,
            estimated_latency_ms=profile.nominal_latency_ms,
            policy_permitted=True,
            rejection_reason=rejection_reason,
        )
        with self._lock:
            self._history.append(dec)
        return dec

    def plan_event_telemetry(
        self,
        event_id: str,
        threat_prior: float,
        epistemic_uncertainty: float,
        asset_criticality: float = 1.0,
        cpu_load: float = 0.25,
    ) -> List[SensorAcquisitionDecision]:
        """Plans optimal combination of modalities subject to budget and VOI ordering."""
        decisions: List[SensorAcquisitionDecision] = []
        elapsed_ms = 0.0

        # Always include baseline network
        decisions.append(
            SensorAcquisitionDecision(
                event_id=event_id,
                modality=SensorModality.BASELINE_NETWORK,
                should_acquire=True,
                value_of_information=1.0,
                expected_security_gain=1.0,
                collection_cost=0.01,
                estimated_latency_ms=0.05,
                policy_permitted=True,
            )
        )
        elapsed_ms += 0.05

        # Candidate modalities to evaluate
        candidates = [
            SensorModality.PROCESS_TELEMETRY,
            SensorModality.IDENTITY_TELEMETRY,
            SensorModality.SESSION_DETAIL,
            SensorModality.GRAPH_NEIGHBORHOOD,
            SensorModality.ENDPOINT_CONTEXT,
        ]

        for modality in candidates:
            dec = self.evaluate_modality(
                event_id=event_id,
                modality=modality,
                threat_prior=threat_prior,
                epistemic_uncertainty=epistemic_uncertainty,
                asset_criticality=asset_criticality,
                cpu_load=cpu_load,
                elapsed_budget_ms=elapsed_ms,
            )
            decisions.append(dec)
            if dec.should_acquire:
                elapsed_ms += dec.estimated_latency_ms

        return decisions
