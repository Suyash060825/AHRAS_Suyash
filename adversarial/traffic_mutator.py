"""
AHRAS Traffic Adversarial Mutator
---------------------------------
Generates semantics-preserving adversarial mutations on network flow telemetry:
- Packet size jitter and padding (+/- Delta s)
- Inter-arrival timing jitter (Poisson/Gaussian noise disrupting periodic autocorrelation)
- Flow duration dilation / rate throttling (dropping PPS below volumetric flood thresholds)
- Packet fragmentation (splitting large flows into smaller MTU transactions)
- Port and protocol variation (e.g. port 80 -> 8080, TLS 443 -> 8443)
- Decoy traffic insertion
"""

from __future__ import annotations

import copy
import enum
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TrafficMutationStrategy(str, enum.Enum):
    PACKET_PADDING = "packet_padding"
    TIMING_JITTER = "timing_jitter"
    DURATION_DILATION = "duration_dilation"
    PACKET_FRAGMENTATION = "packet_fragmentation"
    PORT_VARIATION = "port_variation"
    DECOY_INSERTION = "decoy_insertion"


@dataclass
class TrafficMutationResult:
    strategy: TrafficMutationStrategy
    original_event: Dict[str, Any]
    mutated_event: Dict[str, Any]
    mutation_delta: Dict[str, Any]
    semantic_integrity_preserved: bool = True


class TrafficMutator:
    """
    Applies domain-valid, semantics-preserving mutations to network flow events.
    """
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def mutate_event(
        self,
        event: Dict[str, Any],
        strategy: TrafficMutationStrategy,
        intensity: float = 0.50,
    ) -> TrafficMutationResult:
        """Applies a specific network mutation strategy with controlled perturbation budget."""
        mut_evt = copy.deepcopy(event)
        delta: Dict[str, Any] = {}

        if strategy == TrafficMutationStrategy.PACKET_PADDING:
            # Pad byte sizes or packet lengths
            if "traffic" in mut_evt and "bytes" in mut_evt["traffic"]:
                orig_b = mut_evt["traffic"]["bytes"]
                padded_b = int(orig_b * (1.0 + 0.35 * intensity))
                mut_evt["traffic"]["bytes"] = padded_b
                delta["bytes"] = f"{orig_b} -> {padded_b}"
            if "packet_lengths" in mut_evt:
                orig_lens = mut_evt["packet_lengths"]
                mut_lens = [int(l + 32 * intensity) for l in orig_lens]
                mut_evt["packet_lengths"] = mut_lens
                delta["packet_lengths"] = "padded"

        elif strategy == TrafficMutationStrategy.TIMING_JITTER:
            # Inject timing noise to break periodic beacon autocorrelation
            if "traffic" in mut_evt:
                orig_var = mut_evt["traffic"].get("interval_variance", 0.01)
                new_var = round(orig_var + 0.35 * intensity, 4)
                mut_evt["traffic"]["interval_variance"] = new_var
                delta["interval_variance"] = f"{orig_var} -> {new_var}"
            if "inter_arrival_times" in mut_evt:
                orig_iats = mut_evt["inter_arrival_times"]
                # Add uniform timing jitter across seconds to disrupt periodicity
                mut_iats = [max(0.05, round(t + self.rng.uniform(1.0, 12.0) * intensity, 3)) for t in orig_iats]
                mut_evt["inter_arrival_times"] = mut_iats
                delta["inter_arrival_times"] = "jittered"

        elif strategy == TrafficMutationStrategy.DURATION_DILATION:
            # Stretch session duration and throttle PPS below volumetric flood thresholds
            if "traffic" in mut_evt:
                orig_pkts = mut_evt["traffic"].get("packets", 100)
                orig_dur = mut_evt["traffic"].get("duration_sec", 1.0)
                orig_pps = mut_evt["traffic"].get("pps", orig_pkts / max(0.001, orig_dur))
                new_pps = max(5.0, round(orig_pps * max(0.05, (1.0 - 0.90 * intensity)), 2))
                new_dur = round(orig_pkts / max(0.001, new_pps), 2)
                mut_evt["traffic"]["duration_sec"] = new_dur
                mut_evt["traffic"]["pps"] = new_pps
                delta["duration_sec"] = f"{orig_dur}s -> {new_dur}s"
                delta["pps"] = f"{orig_pps} -> {new_pps} pps"

        elif strategy == TrafficMutationStrategy.PACKET_FRAGMENTATION:
            # Increase packet count by splitting payloads into smaller fragments
            if "traffic" in mut_evt:
                orig_pkts = mut_evt["traffic"].get("packets", 50)
                new_pkts = int(orig_pkts * (1.5 + intensity))
                mut_evt["traffic"]["packets"] = new_pkts
                delta["packets"] = f"{orig_pkts} -> {new_pkts}"

        elif strategy == TrafficMutationStrategy.PORT_VARIATION:
            # Shift standard ports to non-standard alternatives
            dst = mut_evt.get("dst_endpoint", {})
            orig_port = dst.get("port", 80)
            port_map = {80: 8080, 443: 8443, 22: 2222, 21: 2121, 3389: 33890}
            new_port = port_map.get(orig_port, orig_port + 1000)
            if "dst_endpoint" in mut_evt:
                mut_evt["dst_endpoint"]["port"] = new_port
                delta["dst_port"] = f"{orig_port} -> {new_port}"

        elif strategy == TrafficMutationStrategy.DECOY_INSERTION:
            # Add decoy flag or interleave traffic
            if "traffic" in mut_evt:
                orig_pkts = mut_evt["traffic"].get("packets", 100)
                mut_evt["traffic"]["packets"] = int(orig_pkts * 1.2)
                delta["decoy_packets_added"] = True

        return TrafficMutationResult(
            strategy=strategy,
            original_event=event,
            mutated_event=mut_evt,
            mutation_delta=delta,
            semantic_integrity_preserved=True,
        )

    def generate_all_mutations(self, event: Dict[str, Any]) -> List[TrafficMutationResult]:
        """Generates all applicable traffic mutations for an event."""
        results = []
        for strategy in TrafficMutationStrategy:
            res = self.mutate_event(event, strategy)
            if res.mutation_delta:
                results.append(res)
        return results
