"""
Telemetry Fault Injection Engine for AHRAS.
Evaluates detector and orchestrator robustness under real-world sensor degradation,
packet drop, jitter, clock skew, and observation uncertainty.
"""

import random
import copy
from typing import List, Dict, Any

class TelemetryFaultInjector:
    """Injects controlled network & sensor faults into event streams."""

    def __init__(self, fault_rate: float = 0.10, seed: int = 42):
        self.fault_rate = fault_rate
        self.rng = random.Random(seed)

    def inject_event_loss(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drops a fraction of events simulating network drop / sensor buffer overflow."""
        surviving = []
        for e in events:
            if self.rng.random() >= self.fault_rate:
                surviving.append(copy.deepcopy(e))
        return surviving

    def inject_field_corruption(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Corrupts non-critical metadata or sets missing telemetry values."""
        corrupted = []
        for e in events:
            item = copy.deepcopy(e)
            if self.rng.random() < self.fault_rate:
                # Mark observation confidence degradation
                item["observation_confidence"] = max(0.1, 1.0 - self.fault_rate * 1.5)
                # Nullify or scramble a field
                if "bytes_in" in item:
                    item["bytes_in"] = None
                if "port" in item:
                    item["port"] = -1
            corrupted.append(item)
        return corrupted

    def inject_timestamp_skew_and_disorder(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulates sensor clock drift and out-of-order event delivery."""
        skewed = copy.deepcopy(events)
        for e in skewed:
            if self.rng.random() < self.fault_rate:
                skew_sec = self.rng.uniform(-15.0, 15.0)
                if "timestamp" in e and isinstance(e["timestamp"], (int, float)):
                    e["timestamp"] += skew_sec
        # Randomly shuffle a subset to simulate network arrival disorder
        if len(skewed) > 2 and self.fault_rate > 0.05:
            swap_idx = self.rng.randint(0, len(skewed) - 2)
            skewed[swap_idx], skewed[swap_idx + 1] = skewed[swap_idx + 1], skewed[swap_idx]
        return skewed

    def inject_all_faults(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        step1 = self.inject_event_loss(events)
        step2 = self.inject_field_corruption(step1)
        step3 = self.inject_timestamp_skew_and_disorder(step2)
        return step3
