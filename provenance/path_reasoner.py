"""
AHRAS Relational Path Reasoner
------------------------------
Discovers and scores multi-hop attack propagation paths in provenance DAGs,
evaluating cumulative Noisy-OR path risk, kill-chain progression velocity,
missing step reasoning, and critical containment choke-points.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from provenance.models import (
    ProvenanceNodeType,
    ProvenanceEdgeType,
    ProvenanceNode,
    ProvenanceEdge,
)
from provenance.graph import ProvenanceGraph


@dataclass
class AttackPathSummary:
    """
    Evaluated multi-hop attack progression path.
    """
    path_id: str
    nodes: List[str]
    node_types: List[str]
    edges: List[str]
    hop_count: int
    noisy_or_risk: float
    duration_s: float
    velocity_hops_per_hr: float
    critical_node: str
    missing_steps_inferred: int
    stages_detected: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "nodes": self.nodes,
            "node_types": self.node_types,
            "edges": self.edges,
            "hop_count": self.hop_count,
            "noisy_or_risk": round(self.noisy_or_risk, 4),
            "duration_s": round(self.duration_s, 2),
            "velocity_hops_per_hr": round(self.velocity_hops_per_hr, 2),
            "critical_node": self.critical_node,
            "missing_steps_inferred": self.missing_steps_inferred,
            "stages_detected": self.stages_detected,
        }


class RelationalPathReasoner:
    """
    Multi-hop attack path reasoning, Noisy-OR composite risk aggregation,
    and containment choke-point identification.
    """
    def __init__(
        self,
        min_path_risk: float = 0.40,
        max_search_depth: int = 8,
    ) -> None:
        self.min_path_risk = min_path_risk
        self.max_search_depth = max_search_depth

    def compute_noisy_or_risk(self, node_risks: List[float]) -> float:
        """
        Computes probabilistic Noisy-OR composite threat:
        R_path = 1 - prod_{i} (1 - R(v_i))
        """
        prod_complement = 1.0
        for r in node_risks:
            clamped_r = min(0.999, max(0.001, r))
            prod_complement *= (1.0 - clamped_r)
        return float(1.0 - prod_complement)

    def find_attack_paths(
        self,
        graph: ProvenanceGraph,
        source_nodes: Optional[List[str]] = None,
        target_nodes: Optional[List[str]] = None,
    ) -> List[AttackPathSummary]:
        """
        Discovers all directed causal paths from candidate source ingress nodes
        to sensitive targets or alert nodes, evaluating their composite threat.
        """
        # Default sources: External IPs or initial access nodes
        if not source_nodes:
            source_nodes = [
                nid for nid, node in graph.nodes.items()
                if node.node_type == ProvenanceNodeType.IP or len(graph.get_incoming_edges(nid)) == 0
            ]

        # Default targets: Alerts, high-risk assets, or critical files
        if not target_nodes:
            target_nodes = [
                nid for nid, node in graph.nodes.items()
                if node.node_type in {ProvenanceNodeType.ALERT, ProvenanceNodeType.INCIDENT}
                or node.risk_score >= 0.70
            ]

        target_set = set(target_nodes)
        discovered_paths: List[AttackPathSummary] = []
        path_idx = 0

        for src in source_nodes:
            src_node = graph.get_node(src)
            src_risk = src_node.risk_score if src_node else 0.1
            queue = deque([(src, [src], [], [src_risk])])

            while queue:
                curr, curr_nodes, curr_edges, curr_risks = queue.popleft()

                if curr in target_set and len(curr_nodes) > 1:
                    path_idx += 1
                    noisy_or = self.compute_noisy_or_risk(curr_risks)
                    if noisy_or >= self.min_path_risk:
                        # Compute time bounds
                        t_start = graph.nodes[curr_nodes[0]].first_seen
                        t_end = graph.nodes[curr_nodes[-1]].last_seen
                        duration = max(0.1, t_end - t_start)
                        velocity = len(curr_nodes) / max(0.001, duration / 3600.0)

                        # Find highest individual risk node
                        max_risk_node = curr_nodes[int(max(range(len(curr_risks)), key=lambda i: curr_risks[i]))]

                        # Detect unobserved missing steps (e.g. gaps in edge confidence or timestamp jumps > 4h)
                        missing_gaps = 0
                        for i in range(len(curr_nodes) - 1):
                            t1 = graph.nodes[curr_nodes[i]].last_seen
                            t2 = graph.nodes[curr_nodes[i+1]].first_seen
                            if (t2 - t1) > 14400.0:  # 4 hour gap without intermediary
                                missing_gaps += 1

                        # Infer kill chain stages
                        stages = self._infer_kill_chain_stages(curr_nodes, graph)

                        summary = AttackPathSummary(
                            path_id=f"path-{path_idx:03d}",
                            nodes=curr_nodes,
                            node_types=[graph.nodes[n].node_type.value for n in curr_nodes if n in graph.nodes],
                            edges=curr_edges,
                            hop_count=len(curr_edges),
                            noisy_or_risk=noisy_or,
                            duration_s=duration,
                            velocity_hops_per_hr=velocity,
                            critical_node=max_risk_node,
                            missing_steps_inferred=missing_gaps,
                            stages_detected=stages,
                        )
                        discovered_paths.append(summary)

                if len(curr_nodes) >= self.max_search_depth:
                    continue

                for edge in graph.get_outgoing_edges(curr):
                    next_node = edge.target
                    if next_node not in curr_nodes:  # Avoid simple cycles
                        next_risk = graph.nodes[next_node].risk_score if next_node in graph.nodes else 0.1
                        queue.append((
                            next_node,
                            curr_nodes + [next_node],
                            curr_edges + [edge.edge_id],
                            curr_risks + [next_risk],
                        ))

        # Sort paths by risk descending
        discovered_paths.sort(key=lambda p: p.noisy_or_risk, reverse=True)
        return discovered_paths

    def _infer_kill_chain_stages(self, nodes: List[str], graph: ProvenanceGraph) -> List[str]:
        stages = []
        for nid in nodes:
            node = graph.get_node(nid)
            if not node:
                continue
            ntype = node.node_type
            if ntype == ProvenanceNodeType.IP and "RECON_INGRESS" not in stages:
                stages.append("RECON_INGRESS")
            elif ntype == ProvenanceNodeType.PROCESS and "EXECUTION" not in stages:
                stages.append("EXECUTION")
            elif ntype == ProvenanceNodeType.USER and "CREDENTIAL_USE" not in stages:
                stages.append("CREDENTIAL_USE")
            elif ntype == ProvenanceNodeType.FILE and "PERSISTENCE_TAMPERING" not in stages:
                stages.append("PERSISTENCE_TAMPERING")
            elif ntype == ProvenanceNodeType.ALERT and "DETECTION_TRIGGER" not in stages:
                stages.append("DETECTION_TRIGGER")
        return stages

    def identify_critical_chokepoints(
        self,
        paths: List[AttackPathSummary],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Identifies critical bridge nodes that intersect the highest cumulative path risk.
        Interdicting these nodes maximally disrupts attacker lateral movement.
        """
        node_threat_scores: Dict[str, float] = defaultdict(float)

        for p in paths:
            # Exclude endpoints (pure source / target) to find intermediate containment chokepoints
            intermediates = p.nodes[1:-1] if len(p.nodes) > 2 else p.nodes
            for nid in intermediates:
                node_threat_scores[nid] += p.noisy_or_risk

        ranked = sorted(node_threat_scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
