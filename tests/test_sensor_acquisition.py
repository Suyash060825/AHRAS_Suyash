"""
Tests for Adaptive Sensor Acquisition Engine (Section 18 / Research Frontier P1)
---------------------------------------------------------------------------------
Validates:
  1. Default profiles and modality cost/latency/gain ordering.
  2. Value of Information (VOI) calculation with threat prior, uncertainty, and asset criticality.
  3. Cost scaling under congestion and CPU pressure.
  4. Policy gating: strict prohibition of unapproved expensive modalities (e.g. DEEP_FORENSIC).
  5. Latency budget exhaustion prevention.
  6. Multi-modality planning sequence for incoming events.
"""

import pytest
from sensors.sensor_acquisition import (
    SensorModality,
    SensorModalityProfile,
    DEFAULT_SENSOR_PROFILES,
    AcquisitionPolicy,
    SensorAcquisitionDecision,
    AdaptiveSensorAcquisitionEngine,
)


def test_default_profiles_exist_and_monotonic_tradeoffs():
    profiles = DEFAULT_SENSOR_PROFILES
    assert SensorModality.BASELINE_NETWORK in profiles
    assert SensorModality.PROCESS_TELEMETRY in profiles
    assert SensorModality.DEEP_FORENSIC in profiles

    # Baseline network must be cheap and fast
    base = profiles[SensorModality.BASELINE_NETWORK]
    assert base.nominal_latency_ms < 0.1
    assert base.nominal_cost_units < 0.1

    # Deep forensic must have highest information gain but highest cost
    forensic = profiles[SensorModality.DEEP_FORENSIC]
    assert forensic.base_information_gain > 0.90
    assert forensic.nominal_cost_units > 5.0
    assert forensic.requires_explicit_policy is True


def test_expected_security_gain_scaling():
    engine = AdaptiveSensorAcquisitionEngine()

    # Gain on routine benign event with low uncertainty
    low_gain = engine.compute_expected_security_gain(
        modality=SensorModality.PROCESS_TELEMETRY,
        event_threat_prior=0.05,
        epistemic_uncertainty=0.10,
        asset_criticality=1.0,
    )

    # Gain on ambiguous, high-uncertainty threat on critical domain controller
    high_gain = engine.compute_expected_security_gain(
        modality=SensorModality.PROCESS_TELEMETRY,
        event_threat_prior=0.75,
        epistemic_uncertainty=0.85,
        asset_criticality=2.0,
    )

    assert high_gain > low_gain * 2.0


def test_effective_cost_congestion_scaling():
    engine = AdaptiveSensorAcquisitionEngine()

    cost_idle = engine.compute_effective_cost(
        modality=SensorModality.GRAPH_NEIGHBORHOOD,
        cpu_load_factor=0.2,
        queue_latency_factor=0.1,
    )

    cost_congested = engine.compute_effective_cost(
        modality=SensorModality.GRAPH_NEIGHBORHOOD,
        cpu_load_factor=0.9,
        queue_latency_factor=0.85,
    )

    assert cost_congested > cost_idle * 1.5


def test_policy_enforcement_denies_unauthorized_deep_forensics():
    # Policy with deep forensics strictly disabled
    policy = AcquisitionPolicy(allow_deep_forensics=False)
    engine = AdaptiveSensorAcquisitionEngine(policy=policy)

    dec = engine.evaluate_modality(
        event_id="evt-100",
        modality=SensorModality.DEEP_FORENSIC,
        threat_prior=0.99,
        epistemic_uncertainty=0.95,
        asset_criticality=2.0,
    )

    assert dec.should_acquire is False
    assert dec.policy_permitted is False
    assert "POLICY_DENIED_DEEP_FORENSIC" in dec.rejection_reason


def test_latency_budget_exhaustion_guards():
    policy = AcquisitionPolicy(max_total_collection_latency_ms=10.0)
    engine = AdaptiveSensorAcquisitionEngine(policy=policy)

    # Modality requiring 15ms should be rejected when total budget is 10ms
    dec = engine.evaluate_modality(
        event_id="evt-101",
        modality=SensorModality.ENDPOINT_CONTEXT,  # 15ms nominal
        threat_prior=0.8,
        epistemic_uncertainty=0.8,
        elapsed_budget_ms=0.0,
    )

    assert dec.should_acquire is False
    assert dec.rejection_reason == "EXCEEDS_LATENCY_BUDGET"


def test_plan_event_telemetry_workflow():
    engine = AdaptiveSensorAcquisitionEngine()
    plan = engine.plan_event_telemetry(
        event_id="evt-200",
        threat_prior=0.70,
        epistemic_uncertainty=0.65,
        asset_criticality=1.5,
        cpu_load=0.25,
    )

    assert len(plan) > 0
    # First modality must always be baseline network and acquired
    assert plan[0].modality == SensorModality.BASELINE_NETWORK
    assert plan[0].should_acquire is True

    # At least one higher-tier modality should be acquired given the threat prior
    acquired_modalities = [d.modality for d in plan if d.should_acquire]
    assert len(acquired_modalities) >= 2
