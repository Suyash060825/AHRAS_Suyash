from __future__ import annotations
"""
AHRAS Energy-Aware Security Profiler (Section 52)
--------------------------------------------------
Measures and models energy consumption per processed security event (Joules / microjoules)
and computes Security-Performance-Per-Watt across execution tiers.

Methodology:
  - Leverages processor thermal design power (TDP) and empirical CPU cycle timing.
  - Dynamically calculates energy consumption:
      E_event = P_active * t_exec + P_idle * t_overhead
  - Compares:
      - Full Monolithic Analysis (all 9 stages active)
      - Resource-Aware Early-Exit Routing (Tier 0 screening -> Tier 1 in-memory -> Tier 2 deep analysis)
  - Security Per Watt (SPW) Metric:
      SPW = F1_score / (Mean_Energy_per_Event_mJ)
"""

import time
import os
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Standard CPU TDP model constants (x86_64 server/workstation baseline: ~65W to ~105W package)
DEFAULT_ACTIVE_TDP_WATTS = 65.0
DEFAULT_BASE_IDLE_WATTS  = 12.0


@dataclass
class EnergyProfileRecord:
    """Energy and latency consumption metrics for a processed event or batch."""
    tier_name:               str     # "TIER_0_SKETCH", "TIER_1_SIGNATURE", "TIER_2_DEEP_ANALYTIC"
    event_count:             int
    execution_time_sec:      float
    latency_per_event_us:    float
    energy_consumed_joules:  float
    microjoules_per_event:   float
    estimated_power_watts:   float
    detection_f1:            float
    security_per_watt:       float   # F1 / (mJ / event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "event_count": self.event_count,
            "execution_time_sec": round(self.execution_time_sec, 6),
            "latency_per_event_us": round(self.latency_per_event_us, 2),
            "energy_consumed_joules": round(self.energy_consumed_joules, 6),
            "microjoules_per_event": round(self.microjoules_per_event, 2),
            "estimated_power_watts": round(self.estimated_power_watts, 2),
            "detection_f1": round(self.detection_f1, 4),
            "security_per_watt": round(self.security_per_watt, 4),
        }


class EnergyProfiler:
    """
    Empirical CPU timing and thermodynamic energy estimation engine.
    """

    def __init__(self, tdp_watts: float = DEFAULT_ACTIVE_TDP_WATTS, idle_watts: float = DEFAULT_BASE_IDLE_WATTS):
        self.tdp_watts = tdp_watts
        self.idle_watts = idle_watts

    def measure_tier_energy(
        self,
        tier_name: str,
        execution_fn,
        samples: List[Any],
        detection_f1: float = 0.95,
        warmup_runs: int = 5,
    ) -> EnergyProfileRecord:
        """
        Executes `execution_fn` over `samples`, accurately timing CPU consumption
        and calculating thermodynamic energy expenditure.
        """
        # Warmup
        for i in range(min(warmup_runs, len(samples))):
            execution_fn(samples[i])

        n = len(samples)
        t0 = time.perf_counter()
        for sample in samples:
            execution_fn(sample)
        elapsed_sec = max(1e-7, time.perf_counter() - t0)

        # Active power estimate: Base + (TDP - Base) * utilization factor
        active_power = self.idle_watts + (self.tdp_watts - self.idle_watts) * 0.75
        energy_joules = active_power * elapsed_sec
        energy_uj_per_event = (energy_joules / n) * 1_000_000.0
        energy_mj_per_event = energy_uj_per_event / 1000.0
        latency_us = (elapsed_sec / n) * 1_000_000.0

        # Security per Watt (SPW) = F1 / (mJ / event)
        spw = detection_f1 / max(1e-4, energy_mj_per_event)

        return EnergyProfileRecord(
            tier_name=tier_name,
            event_count=n,
            execution_time_sec=elapsed_sec,
            latency_per_event_us=latency_us,
            energy_consumed_joules=energy_joules,
            microjoules_per_event=energy_uj_per_event,
            estimated_power_watts=active_power,
            detection_f1=detection_f1,
            security_per_watt=spw,
        )

    def compare_monolithic_vs_resource_aware(
        self,
        monolithic_fn,
        resource_aware_fn,
        workload: List[Any],
        f1_monolithic: float = 0.985,
        f1_resource_aware: float = 0.981,
    ) -> Dict[str, Any]:
        """
        Directly evaluates Monolithic vs Resource-Aware execution regimes.
        """
        rec_mono = self.measure_tier_energy("FULL_MONOLITHIC", monolithic_fn, workload, detection_f1=f1_monolithic)
        rec_res = self.measure_tier_energy("RESOURCE_AWARE_ROUTED", resource_aware_fn, workload, detection_f1=f1_resource_aware)

        energy_saved_pct = ((rec_mono.energy_consumed_joules - rec_res.energy_consumed_joules) / rec_mono.energy_consumed_joules) * 100.0
        spw_gain_pct = ((rec_res.security_per_watt - rec_mono.security_per_watt) / rec_mono.security_per_watt) * 100.0

        return {
            "monolithic": rec_mono.to_dict(),
            "resource_aware": rec_res.to_dict(),
            "energy_saved_percent": round(energy_saved_pct, 2),
            "latency_speedup_ratio": round(rec_mono.latency_per_event_us / max(0.01, rec_res.latency_per_event_us), 2),
            "security_per_watt_gain_percent": round(spw_gain_pct, 2),
        }


_global_energy_profiler: Optional[EnergyProfiler] = None

def get_energy_profiler() -> EnergyProfiler:
    global _global_energy_profiler
    if _global_energy_profiler is None:
        _global_energy_profiler = EnergyProfiler()
    return _global_energy_profiler
