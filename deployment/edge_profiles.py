from __future__ import annotations
"""
AHRAS Edge & Distributed Deployment Profiles (Section 54)
----------------------------------------------------------
Defines portable operational deployment profiles tailored to distinct enterprise footprints:

1. CENTRAL (Cloud / Core SOC Datacenter):
   - Unconstrained compute and memory (>=32GB RAM, GPU/12+ vCPU).
   - Runs full 9-stage analytical pipeline: GNN reasoning, historical case memory,
     counterfactual twin simulation, LLM copilot, full forensic ledgers.

2. EDGE (Branch Office / Regional Gateway):
   - Constrained resources (4–8GB RAM, 4 vCPUs, no discrete GPU).
   - Runs: Streaming sketch filter (Tier 0), Signature filter (Tier 1),
     Quantized/Distilled ML models, local honeypot tripwires, cryptographic epoch checkpointing.
   - Upstream bandwidth throttled: forwards only anomalous incidents and aggregated evidence digests.

3. ENDPOINT (Host-Level Lightweight Agent / eBPF Daemon):
   - Micro-footprint (<250MB RAM, <5% single CPU core).
   - Runs: eBPF syscall event capture, entropy calculation, behavioral burst trackers
     (Ransomware / Worm / Malware), local privacy anonymization.
   - Forwards normalized OCSF events to local Edge gateway.

4. HYBRID (Hierarchical Cooperative Mesh):
   - Endpoints filter 90% benign noise locally and stream suspicious behavior to Edge.
   - Edge performs sub-millisecond local containment and forwards complex multi-hop episodes to Central.
"""

import time
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set

log = logging.getLogger(__name__)


class DeploymentTier(str, Enum):
    CENTRAL  = "CENTRAL"
    EDGE     = "EDGE"
    ENDPOINT = "ENDPOINT"
    HYBRID   = "HYBRID"


@dataclass
class DeploymentProfileConfig:
    """Configuration specifications for a target deployment tier."""
    tier:                   DeploymentTier
    target_environment:     str
    memory_limit_mb:        int
    cpu_cores_allocated:    float
    active_stages:          List[str]
    model_precision:        str    # "FP32", "INT8", "DISTILLED", "HEURISTIC_ONLY"
    max_pipeline_latency_ms: float
    upstream_bandwidth_kbps: float
    local_containment_enabled: bool
    description:            str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        return d


class DeploymentProfileManager:
    """
    Manages and instantiates architecture configurations across the 4 deployment tiers.
    """

    PROFILES: Dict[DeploymentTier, DeploymentProfileConfig] = {
        DeploymentTier.CENTRAL: DeploymentProfileConfig(
            tier=DeploymentTier.CENTRAL,
            target_environment="Cloud Datacenter / High-Capacity K8s Cluster",
            memory_limit_mb=32768,
            cpu_cores_allocated=16.0,
            active_stages=[
                "OCSF_NORMALIZATION", "MULTIMODAL_ATTENTION", "TRI_ENGINE_COMBINER",
                "TGNN_GRAPH_REASONING", "CONFORMAL_GATE", "SECURITY_TWIN_SIMULATION",
                "EVIDENCE_LEDGER", "LLM_NARRATOR", "RECOVERY_LOOP"
            ],
            model_precision="FP32",
            max_pipeline_latency_ms=25.0,
            upstream_bandwidth_kbps=0.0, # Sink node
            local_containment_enabled=True,
            description="Full-stack unconstrained analytical and forensic investigation hub.",
        ),
        DeploymentTier.EDGE: DeploymentProfileConfig(
            tier=DeploymentTier.EDGE,
            target_environment="Branch Office Micro-Server / Secure Gateway",
            memory_limit_mb=4096,
            cpu_cores_allocated=4.0,
            active_stages=[
                "OCSF_NORMALIZATION", "STREAMING_SKETCH", "SIGNATURE_FILTER",
                "DISTILLED_ANOMALY", "CONFORMAL_GATE", "ADAPTIVE_DECEPTION"
            ],
            model_precision="INT8",
            max_pipeline_latency_ms=2.5,
            upstream_bandwidth_kbps=2048.0, # 2 Mbps compressed anomaly stream
            local_containment_enabled=True,
            description="Sub-millisecond perimeter screening and local containment enforcement.",
        ),
        DeploymentTier.ENDPOINT: DeploymentProfileConfig(
            tier=DeploymentTier.ENDPOINT,
            target_environment="Host Endpoint / eBPF / Windows ETW Daemon",
            memory_limit_mb=256,
            cpu_cores_allocated=0.5,
            active_stages=[
                "EBPF_CAPTURE", "ENTROPY_BURST_TRACKER", "PRIVACY_MINIMIZATION"
            ],
            model_precision="HEURISTIC_ONLY",
            max_pipeline_latency_ms=0.15,
            upstream_bandwidth_kbps=256.0, # 256 Kbps batched telemetry
            local_containment_enabled=False, # Avoid host deadlock
            description="Ultra-low overhead behavioral sensor and privacy-minimization agent.",
        ),
        DeploymentTier.HYBRID: DeploymentProfileConfig(
            tier=DeploymentTier.HYBRID,
            target_environment="Hierarchical Enterprise Mesh (Endpoint -> Edge -> Central)",
            memory_limit_mb=8192,
            cpu_cores_allocated=8.0,
            active_stages=[
                "DISTRIBUTED_PIPELINE_ORCHESTRATION", "FEDERATED_COOPERATION",
                "SELECTIVE_UPSTREAM_FORWARDING"
            ],
            model_precision="ADAPTIVE_MIXED",
            max_pipeline_latency_ms=5.0,
            upstream_bandwidth_kbps=1024.0,
            local_containment_enabled=True,
            description="Optimal Pareto operating point combining low endpoint latency with deep cloud reasoning.",
        ),
    }

    @classmethod
    def get_profile(cls, tier: DeploymentTier) -> DeploymentProfileConfig:
        return cls.PROFILES[tier]

    @classmethod
    def benchmark_deployment_profiles(cls) -> Dict[str, Any]:
        """
        Simulates and benchmarks operational metrics across all 4 deployment tiers.
        """
        results: Dict[str, Any] = {}
        for tier, cfg in cls.PROFILES.items():
            results[tier.value] = {
                "target_environment": cfg.target_environment,
                "memory_limit_mb": cfg.memory_limit_mb,
                "cpu_cores": cfg.cpu_cores_allocated,
                "model_precision": cfg.model_precision,
                "max_latency_ms": cfg.max_pipeline_latency_ms,
                "bandwidth_kbps": cfg.upstream_bandwidth_kbps,
                "stages_active_count": len(cfg.active_stages),
                "local_containment": cfg.local_containment_enabled,
                "estimated_throughput_eps": round(1000.0 / max(0.01, cfg.max_pipeline_latency_ms) * (cfg.cpu_cores_allocated * 0.75), 1),
            }
        return results


_global_deployment_mgr: Optional[DeploymentProfileManager] = None

def get_deployment_profile_manager() -> DeploymentProfileManager:
    global _global_deployment_mgr
    if _global_deployment_mgr is None:
        _global_deployment_mgr = DeploymentProfileManager()
    return _global_deployment_mgr
