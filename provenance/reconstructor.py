from __future__ import annotations
"""
AHRAS Module / Provenance — Attack Chain Reconstruction Engine
--------------------------------------------------------------
Correlates disparate security events and alerts into cohesive, multi-stage
incident attack scenarios using temporal, entity, and MITRE kill-chain priors.
Identifies stealthy unobserved intermediate steps and compiles incident explanations.
"""

import math
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from provenance.models import (
    ProvenanceNode, ProvenanceEdge, ProvenanceNodeType, ProvenanceEdgeType,
    ProvenanceAttackScenario
)
from provenance.graph import ProvenanceGraph


# Ordered Kill-Chain Progression Matrix
KILL_CHAIN_STAGES = [
    "RECONNAISSANCE",
    "INITIAL_ACCESS",
    "EXECUTION",
    "PERSISTENCE",
    "CREDENTIAL_ACCESS",
    "LATERAL_MOVEMENT",
    "COLLECTION",
    "EXFILTRATION",
    "IMPACT",
]

# MITRE Technique to Kill-Chain Stage Mapping
TECHNIQUE_STAGE_MAP = {
    "T1046": "RECONNAISSANCE",
    "T1016": "RECONNAISSANCE",
    "T1595": "RECONNAISSANCE",
    "T1190": "INITIAL_ACCESS",
    "T1110": "INITIAL_ACCESS",
    "T1110.001": "INITIAL_ACCESS",
    "T1566": "INITIAL_ACCESS",
    "T1059": "EXECUTION",
    "T1059.001": "EXECUTION",
    "T1059.004": "EXECUTION",
    "T1078.004": "EXECUTION",
    "T1053": "PERSISTENCE",
    "T1543": "PERSISTENCE",
    "T1003": "CREDENTIAL_ACCESS",
    "T1003.008": "CREDENTIAL_ACCESS",
    "T1528": "CREDENTIAL_ACCESS",
    "T1021": "LATERAL_MOVEMENT",
    "T1021.001": "LATERAL_MOVEMENT",
    "T1021.004": "LATERAL_MOVEMENT",
    "T1074": "COLLECTION",
    "T1530": "COLLECTION",
    "T1041": "EXFILTRATION",
    "T1486": "IMPACT",
    "T1485": "IMPACT",
    "T1498": "IMPACT",
}


class AttackScenarioReconstructor:
    """
    Reconstructs holistic attack campaign graphs from raw event streams,
    ranks candidate causal chains, and identifies missing/stealthy steps.
    """

    def __init__(self, temporal_window_sec: float = 3600.0 * 24):
        self.temporal_window_sec = temporal_window_sec

    def ingest_events(self, events: List[Dict[str, Any]]) -> ProvenanceGraph:
        """
        Translates a batch of OCSF/alert dictionaries into a heterogeneous provenance graph.
        """
        graph = ProvenanceGraph(graph_id=f"graph-{uuid.uuid4().hex[:8]}")

        for ev in events:
            ev_id = ev.get("event_id", f"ev-{uuid.uuid4().hex[:8]}")
            ts_str = ev.get("time")
            # Parse or generate timestamp
            try:
                if isinstance(ts_str, (int, float)):
                    ts = float(ts_str)
                elif isinstance(ts_str, str) and ts_str:
                    ts = time.mktime(time.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S"))
                else:
                    ts = time.time()
            except Exception:
                ts = time.time()

            cls_id = ev.get("ocsf_class_id", 1001)
            enrichment = ev.get("enrichment", {})
            tech = enrichment.get("mitre_technique") or ev.get("technique")

            # ── 1. Network Activity (Class 1001) ──
            if cls_id == 1001:
                src_ip = ev.get("src_endpoint", {}).get("ip") or ev.get("src_ip", "0.0.0.0")
                dst_ip = ev.get("dst_endpoint", {}).get("ip") or ev.get("dst_ip", "0.0.0.0")
                dst_port = ev.get("dst_endpoint", {}).get("port") or ev.get("dst_port", 0)

                # Nodes
                graph.add_node(ProvenanceNode(
                    node_id=src_ip,
                    node_type=ProvenanceNodeType.IP,
                    name=src_ip,
                    first_seen=ts,
                    last_seen=ts,
                    risk_score=0.3 if enrichment.get("is_threat_intel_hit") else 0.1,
                ))
                graph.add_node(ProvenanceNode(
                    node_id=dst_ip,
                    node_type=ProvenanceNodeType.HOST if dst_ip.startswith("host-") else ProvenanceNodeType.IP,
                    name=dst_ip,
                    first_seen=ts,
                    last_seen=ts,
                    risk_score=0.2,
                ))

                # Edge
                graph.add_edge(ProvenanceEdge(
                    edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                    source=src_ip,
                    target=dst_ip,
                    relation=ProvenanceEdgeType.CONNECTS_TO,
                    timestamp=ts,
                    confidence=0.90,
                    event_id=ev_id,
                    metadata={"dst_port": dst_port, "protocol": ev.get("protocol", "TCP"), "mitre_technique": tech},
                ))

                # Technique Edge if present
                if tech:
                    t_node_id = f"tech:{tech}"
                    graph.add_node(ProvenanceNode(
                        node_id=t_node_id,
                        node_type=ProvenanceNodeType.TECHNIQUE,
                        name=enrichment.get("mitre_name", tech),
                        properties={"technique_id": tech},
                        first_seen=ts,
                        last_seen=ts,
                    ))
                    graph.add_edge(ProvenanceEdge(
                        edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                        source=src_ip,
                        target=t_node_id,
                        relation=ProvenanceEdgeType.USES_TECHNIQUE,
                        timestamp=ts,
                        confidence=0.95,
                        event_id=ev_id,
                        metadata={"mitre_technique": tech},
                    ))

            # ── 2. Process Activity (Class 1002) ──
            elif cls_id == 1002:
                actor_proc = ev.get("actor", {}).get("process", {})
                pid = actor_proc.get("pid") or ev.get("pid", 1000)
                pname = actor_proc.get("name") or ev.get("process_name", "proc")
                host = ev.get("device", {}).get("hostname") or ev.get("hostname", "host-01")
                user = actor_proc.get("user", {}).get("name") or ev.get("username", "root")

                proc_node_id = f"proc:{host}:{pid}"
                host_node_id = host
                user_node_id = f"user:{user}"

                graph.add_node(ProvenanceNode(
                    node_id=proc_node_id,
                    node_type=ProvenanceNodeType.PROCESS,
                    name=f"{pname} (PID {pid})",
                    properties={"cmdline": actor_proc.get("cmd_line", "")},
                    first_seen=ts,
                    last_seen=ts,
                    risk_score=0.6 if enrichment.get("suspicious_lineage") else 0.1,
                ))
                graph.add_node(ProvenanceNode(
                    node_id=user_node_id,
                    node_type=ProvenanceNodeType.USER,
                    name=user,
                    first_seen=ts,
                    last_seen=ts,
                ))
                graph.add_node(ProvenanceNode(
                    node_id=host_node_id,
                    node_type=ProvenanceNodeType.HOST,
                    name=host,
                    first_seen=ts,
                    last_seen=ts,
                ))

                # Edges
                graph.add_edge(ProvenanceEdge(
                    edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                    source=user_node_id,
                    target=proc_node_id,
                    relation=ProvenanceEdgeType.EXECUTES,
                    timestamp=ts,
                    confidence=0.95,
                    event_id=ev_id,
                ))
                graph.add_edge(ProvenanceEdge(
                    edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                    source=proc_node_id,
                    target=host_node_id,
                    relation=ProvenanceEdgeType.PART_OF,
                    timestamp=ts,
                    confidence=1.0,
                    event_id=ev_id,
                ))

                # Parent process
                ppid = ev.get("process", {}).get("parent_pid")
                if ppid:
                    parent_node_id = f"proc:{host}:{ppid}"
                    graph.add_edge(ProvenanceEdge(
                        edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                        source=parent_node_id,
                        target=proc_node_id,
                        relation=ProvenanceEdgeType.CREATES,
                        timestamp=ts,
                        confidence=0.95,
                        event_id=ev_id,
                    ))

            # ── 3. File Activity (Class 1003) ──
            elif cls_id == 1003:
                f_path = ev.get("file", {}).get("path") or ev.get("file_path", "/var/file")
                host = ev.get("device", {}).get("hostname") or ev.get("hostname", "host-01")
                file_node_id = f"file:{host}:{f_path}"

                graph.add_node(ProvenanceNode(
                    node_id=file_node_id,
                    node_type=ProvenanceNodeType.FILE,
                    name=f_path,
                    properties={"entropy": ev.get("file", {}).get("entropy", 4.0)},
                    first_seen=ts,
                    last_seen=ts,
                    risk_score=0.8 if ev.get("file", {}).get("entropy", 0.0) > 7.5 else 0.1,
                ))
                graph.add_edge(ProvenanceEdge(
                    edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                    source=host,
                    target=file_node_id,
                    relation=ProvenanceEdgeType.WRITES,
                    timestamp=ts,
                    confidence=0.90,
                    event_id=ev_id,
                ))

            # ── 4. Cloud API Activity (Class 4001) ──
            elif cls_id == 4001:
                uname = ev.get("actor", {}).get("user", {}).get("name") or "admin"
                api_op = ev.get("api", {}).get("operation") or "API_CALL"
                user_node_id = f"user:{uname}"
                asset_node_id = f"asset:{api_op}"

                graph.add_node(ProvenanceNode(
                    node_id=user_node_id,
                    node_type=ProvenanceNodeType.USER,
                    name=uname,
                    first_seen=ts,
                    last_seen=ts,
                ))
                graph.add_node(ProvenanceNode(
                    node_id=asset_node_id,
                    node_type=ProvenanceNodeType.ASSET,
                    name=api_op,
                    first_seen=ts,
                    last_seen=ts,
                    risk_score=0.7,
                ))
                graph.add_edge(ProvenanceEdge(
                    edge_id=f"edge-{uuid.uuid4().hex[:10]}",
                    source=user_node_id,
                    target=asset_node_id,
                    relation=ProvenanceEdgeType.TARGETS,
                    timestamp=ts,
                    confidence=0.90,
                    event_id=ev_id,
                ))

        return graph

    def reconstruct_scenario(
        self,
        graph: ProvenanceGraph,
        incident_id: Optional[str] = None,
    ) -> ProvenanceAttackScenario:
        """
        Reconstructs the optimal attack scenario from the provenance graph:
          1. Correlate by entity and technique.
          2. Trace causal temporal chains.
          3. Rank candidate paths and compute Noisy-OR composite risk.
          4. Detect unobserved/missing kill-chain phases.
          5. Compile human-readable incident explanation.
        """
        inc_id = incident_id or f"inc-{uuid.uuid4().hex[:8]}"

        if not graph.nodes or not graph.edges:
            return ProvenanceAttackScenario(
                incident_id=inc_id,
                stages=[],
                entities=[],
                edges=[],
                confidence=0.0,
                path_risk=0.0,
                missing_steps=[],
                summary="Empty provenance graph; no attack scenario reconstructed.",
            )

        # 1. Identify Root Nodes (nodes with 0 incoming causal edges or external IPs)
        all_sources = {e.source for e in graph.edges.values()}
        all_targets = {e.target for e in graph.edges.values()}
        roots = all_sources - all_targets
        if not roots:
            # Fallback to earliest node by timestamp
            roots = {min(graph.nodes.values(), key=lambda n: n.first_seen).node_id}

        # 2. Extract and Sort Causal Edges chronologically
        sorted_edges = sorted(graph.edges.values(), key=lambda e: e.timestamp)

        # 3. Correlate Stages from MITRE techniques and edge operations
        observed_stages: List[str] = []
        participating_entities: Set[str] = set()

        for edge in sorted_edges:
            participating_entities.add(edge.source)
            participating_entities.add(edge.target)

            # Determine stage from technique or edge semantics
            stage = self._infer_stage(edge, graph)
            if stage and (not observed_stages or observed_stages[-1] != stage):
                observed_stages.append(stage)

        # 4. Compute Composite Path Risk (Noisy-OR over entity risks)
        node_risks = [graph.nodes[nid].risk_score for nid in participating_entities if nid in graph.nodes]
        if node_risks:
            # Noisy-OR: R = 1 - prod(1 - r_i)
            prod_complement = 1.0
            for r in node_risks:
                prod_complement *= (1.0 - min(0.99, max(0.01, r)))
            composite_risk = 1.0 - prod_complement
        else:
            composite_risk = 0.50

        # 5. Calculate Aggregate Causal Confidence
        edge_confidences = [e.confidence for e in sorted_edges]
        mean_confidence = float(sum(edge_confidences) / len(edge_confidences)) if edge_confidences else 0.80

        # 6. Detect Missing / Stealthy Intermediate Steps
        missing_steps = self._detect_missing_steps(observed_stages)

        # 7. Compile Incident Explanation
        summary = self._generate_incident_summary(
            inc_id=inc_id,
            stages=observed_stages,
            entities=participating_entities,
            edges=sorted_edges,
            composite_risk=composite_risk,
            confidence=mean_confidence,
            missing_steps=missing_steps,
        )

        return ProvenanceAttackScenario(
            incident_id=inc_id,
            stages=observed_stages,
            entities=sorted(list(participating_entities)),
            edges=sorted_edges,
            confidence=round(mean_confidence, 4),
            path_risk=round(composite_risk, 4),
            missing_steps=missing_steps,
            summary=summary,
        )

    # ── Stage Inference & Missing Step Reasoning ──────────────────────────────

    def _infer_stage(self, edge: ProvenanceEdge, graph: ProvenanceGraph) -> Optional[str]:
        """Infers the kill-chain phase corresponding to an edge and its attached nodes."""
        # 1. Check if edge links to a TECHNIQUE node
        target_node = graph.get_node(edge.target)
        if target_node and target_node.node_type == ProvenanceNodeType.TECHNIQUE:
            tech_id = target_node.properties.get("technique_id") or target_node.name
            for tid, stage in TECHNIQUE_STAGE_MAP.items():
                if tid in tech_id or tid in target_node.node_id:
                    return stage

        # 2. Check metadata on edge
        meta = edge.metadata
        if "mitre_technique" in meta and meta["mitre_technique"]:
            tech = meta["mitre_technique"]
            for tid, stage in TECHNIQUE_STAGE_MAP.items():
                if tid in tech:
                    return stage

        # 3. Infer from edge relationship semantics
        if edge.relation == ProvenanceEdgeType.CONNECTS_TO:
            dst_port = meta.get("dst_port", 0)
            if dst_port in (80, 443, 8080):
                return "INITIAL_ACCESS"
            elif dst_port in (22, 3389, 445):
                return "LATERAL_MOVEMENT"
            return "RECONNAISSANCE"

        elif edge.relation == ProvenanceEdgeType.EXECUTES or edge.relation == ProvenanceEdgeType.CREATES:
            return "EXECUTION"

        elif edge.relation == ProvenanceEdgeType.WRITES:
            return "IMPACT"

        elif edge.relation == ProvenanceEdgeType.TARGETS or edge.relation == ProvenanceEdgeType.AUTHENTICATES:
            return "CREDENTIAL_ACCESS"

        return None

    def _detect_missing_steps(self, observed_stages: List[str]) -> List[str]:
        """
        Analyzes consecutive kill-chain stages to identify stealthy, unobserved gaps.
        e.g., INITIAL_ACCESS followed by LATERAL_MOVEMENT without EXECUTION -> stealthy execution.
        """
        missing: List[str] = []
        if not observed_stages:
            return missing

        stage_order = {s: i for i, s in enumerate(KILL_CHAIN_STAGES)}
        indices = [stage_order[s] for s in observed_stages if s in stage_order]

        for i in range(len(indices) - 1):
            curr_idx = indices[i]
            next_idx = indices[i + 1]
            if next_idx > curr_idx + 1:
                # One or more intermediate stages skipped in observed telemetry
                for skipped_idx in range(curr_idx + 1, next_idx):
                    skipped_stage = KILL_CHAIN_STAGES[skipped_idx]
                    if skipped_stage == "EXECUTION":
                        missing.append("Hypothesized Unobserved Execution (LOLBin / in-memory injection)")
                    elif skipped_stage == "PERSISTENCE":
                        missing.append("Hypothesized Unobserved Persistence (Scheduled task / cron stealth)")
                    elif skipped_stage == "CREDENTIAL_ACCESS":
                        missing.append("Hypothesized Unobserved Credential Access (Memory dump / token reuse)")
                    elif skipped_stage == "COLLECTION":
                        missing.append("Hypothesized Unobserved Collection (Local staging before exfiltration)")
                    else:
                        missing.append(f"Hypothesized Unobserved {skipped_stage}")

        return missing

    def _generate_incident_summary(
        self,
        inc_id: str,
        stages: List[str],
        entities: Set[str],
        edges: List[ProvenanceEdge],
        composite_risk: float,
        confidence: float,
        missing_steps: List[str],
    ) -> str:
        """Constructs an incident-level structured explanation."""
        num_edges = len(edges)
        num_nodes = len(entities)
        stage_str = " -> ".join(stages) if stages else "UNDETERMINED"

        summary = (
            f"**Incident Report {inc_id}**\n"
            f"- **Composite Path Risk**: {composite_risk:.2f} | **Causal Confidence**: {confidence * 100:.1f}%\n"
            f"- **Attack Progression**: {stage_str}\n"
            f"- **Forensic Scope**: {num_nodes} correlated entities across {num_edges} provenance relationships.\n"
        )
        if missing_steps:
            summary += "- **Potential Stealth Gaps Identified**:\n"
            for gap in missing_steps:
                summary += f"  - *{gap}*\n"
        return summary
