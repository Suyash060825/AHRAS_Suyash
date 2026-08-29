from __future__ import annotations
"""
AHRAS Module 6 — Attack Episode & Temporal Evidence Graph Engine
----------------------------------------------------------------
Constructs an evolving temporal graph across heterogeneous security entities:
  - Source IP, Destination IP, Hostname, Username, Process, File, Cloud IAM, IOC, MITRE Technique.

Key Research Capabilities:
  1. Attack Episode Reconstruction: Groups temporally coherent multi-entity actions into discrete intrusion episodes.
  2. Multi-Hop Lateral Movement Detection: Identifies anomalous traversal chains (T1021, T1078, T1059).
  3. Contextual Corroboration Modulation: Evaluates graph centrality & evidence density to modulate risk confidence.
"""

import math
import time
import uuid
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GraphEdge:
    source:        str
    target:        str
    relation:      str            # CONNECTS_TO, SPAWNS, ACCESSED, AUTHENTICATED_AS, COMMUNICATES_WITH, TRIGGERED
    first_seen:    float
    last_seen:     float
    confidence:    float = 1.0
    evidence_count: int = 1
    evidence_ids:  List[str] = field(default_factory=list)


@dataclass
class AttackEpisode:
    episode_id:        str
    root_entity:       str
    entities:          List[str]
    edges:             List[Dict[str, Any]]
    start_time:        float
    end_time:          float
    duration_sec:      float
    attack_stages:     List[str]      # e.g. RECONNAISSANCE, LATERAL_MOVEMENT, EXFILTRATION
    mitre_techniques:  List[str]
    corroboration_score: float        # [0.0, 1.0] confidence boost from multi-node corroboration
    is_multi_stage:    bool


@dataclass
class GraphPathAnomaly:
    """Output from scoring multi-hop path anomaly."""
    source_entity:       str
    target_entity:       str
    path_length:         int
    path_nodes:          List[str]
    path_risk_score:     float
    is_lateral_movement: bool
    mitre_techniques:    List[str] = field(default_factory=list)
    episode_id:          Optional[str] = None


class EntityGraphEngine:
    """
    Maintains a multi-hop temporal entity graph and detects anomalous lateral movement paths.
    Thread-safe via RLock.
    """

    def __init__(self, max_hops: int = 5, episode_window_sec: float = 3600.0):
        # Graph adjacency: node -> {target_node: GraphEdge}
        self._adj: Dict[str, Dict[str, GraphEdge]] = defaultdict(dict)
        # Node access frequencies: node -> count
        self._node_freq: Dict[str, int] = defaultdict(int)
        self._node_types: Dict[str, str] = {}
        self._max_hops = max_hops
        self._episode_window = episode_window_sec
        self._lock = threading.RLock()

    def add_event_edge(
        self,
        src_node: str,
        dst_node: str,
        relation: str,
        ts: Optional[float] = None,
        evidence_id: Optional[str] = None,
        confidence: float = 1.0,
        src_type: str = "host",
        dst_type: str = "resource",
    ) -> None:
        """Adds or updates a directed temporal edge between two entities."""
        if not src_node or not dst_node or src_node == dst_node:
            return
        if ts is None:
            ts = time.time()

        with self._lock:
            self._node_types[src_node] = src_type
            self._node_types[dst_node] = dst_type
            self._node_freq[src_node] += 1
            self._node_freq[dst_node] += 1

            edge = self._adj[src_node].get(dst_node)
            if edge is None:
                edge = GraphEdge(
                    source=src_node,
                    target=dst_node,
                    relation=relation,
                    first_seen=ts,
                    last_seen=ts,
                    confidence=confidence,
                    evidence_count=1,
                    evidence_ids=[evidence_id] if evidence_id else [],
                )
                self._adj[src_node][dst_node] = edge
            else:
                edge.last_seen = max(edge.last_seen, ts)
                edge.evidence_count += 1
                if evidence_id and evidence_id not in edge.evidence_ids:
                    edge.evidence_ids.append(evidence_id)

    def analyze_lateral_movement(self, src_node: str, dst_node: str) -> GraphPathAnomaly:
        """
        Calculates shortest path and evaluates lateral movement path anomaly score.
        """
        with self._lock:
            path = self._find_shortest_path(src_node, dst_node)
            dst_freq = self._node_freq[dst_node]

        if not path:
            return GraphPathAnomaly(
                source_entity=src_node, target_entity=dst_node, path_length=0,
                path_nodes=[], path_risk_score=0.0, is_lateral_movement=False
            )

        path_len = len(path) - 1
        rarity_score = 1.0 / math.sqrt(max(dst_freq, 1))
        path_risk = min((path_len * 0.25) + (rarity_score * 0.5), 1.0)
        path_risk = round(float(path_risk), 4)

        is_anom = path_len >= 3 or path_risk >= 0.70
        mitre = []
        if is_anom:
            mitre.append("T1021")  # Remote Services / Lateral Movement
            mitre.append("T1078")  # Valid Accounts
            mitre.append("T1059")  # Command and Scripting Interpreter

        return GraphPathAnomaly(
            source_entity=src_node,
            target_entity=dst_node,
            path_length=path_len,
            path_nodes=path,
            path_risk_score=path_risk,
            is_lateral_movement=is_anom,
            mitre_techniques=mitre,
        )

    def get_corroboration_score(self, entity_id: str) -> float:
        """
        Computes an episode corroboration multiplier [0.0, 1.0] based on connectivity degree.
        Single isolated anomalies get 0.0; multi-connected entities get up to 1.0.
        """
        with self._lock:
            out_degree = len(self._adj.get(entity_id, {}))
            in_degree = sum(1 for s in self._adj if entity_id in self._adj[s])
            total_conn = out_degree + in_degree
            if total_conn <= 1:
                return 0.0
            return round(min(1.0, (total_conn - 1) * 0.25), 3)

    def build_attack_episode(self, root_entity: str, max_depth: int = 4) -> Optional[AttackEpisode]:
        """
        Constructs an AttackEpisode traversing outgoing and incoming relations from root_entity.
        """
        with self._lock:
            if root_entity not in self._adj and not any(root_entity in self._adj[s] for s in self._adj):
                return None

            visited: Set[str] = {root_entity}
            queue = deque([(root_entity, 0)])
            collected_edges: List[Dict[str, Any]] = []
            min_time = float("inf")
            max_time = 0.0
            techniques: Set[str] = set()

            while queue:
                curr, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                for neighbor, edge in self._adj.get(curr, {}).items():
                    min_time = min(min_time, edge.first_seen)
                    max_time = max(max_time, edge.last_seen)
                    collected_edges.append({
                        "source": edge.source,
                        "target": edge.target,
                        "relation": edge.relation,
                        "evidence_count": edge.evidence_count,
                    })
                    if edge.relation in ("SSH", "RDP", "SMB"):
                        techniques.add("T1021")
                    elif edge.relation in ("DUMP", "PRIV_ESC"):
                        techniques.add("T1003")

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

            if min_time == float("inf"):
                min_time = time.time()
                max_time = min_time

            duration = max(0.0, max_time - min_time)
            stages = []
            if len(visited) > 1: stages.append("INITIAL_ACCESS")
            if any(e["relation"] in ("SCAN", "PROBE") for e in collected_edges): stages.append("RECONNAISSANCE")
            if any(e["relation"] in ("SSH", "SMB", "RDP") for e in collected_edges): stages.append("LATERAL_MOVEMENT")
            if any(e["relation"] in ("DUMP", "VAULT") for e in collected_edges): stages.append("CREDENTIAL_ACCESS")

            corroboration = min(1.0, len(visited) * 0.20 + len(collected_edges) * 0.10)

            return AttackEpisode(
                episode_id=f"EPISODE-{uuid.uuid4().hex[:8]}",
                root_entity=root_entity,
                entities=list(visited),
                edges=collected_edges,
                start_time=min_time,
                end_time=max_time,
                duration_sec=round(duration, 2),
                attack_stages=stages or ["INITIAL_ACCESS"],
                mitre_techniques=sorted(list(techniques)),
                corroboration_score=round(corroboration, 3),
                is_multi_stage=(len(stages) >= 2),
            )

    def _find_shortest_path(self, start: str, end: str) -> List[str]:
        if start not in self._adj:
            return []

        visited = {start}
        queue = [[start]]

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node == end:
                return path

            if len(path) > self._max_hops:
                continue

            for neighbor in self._adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return []

    def clear(self) -> None:
        with self._lock:
            self._adj.clear()
            self._node_freq.clear()
            self._node_types.clear()


# Singleton
_graph_engine_instance: Optional[EntityGraphEngine] = None
_graph_lock = threading.Lock()


def get_entity_graph_engine() -> EntityGraphEngine:
    global _graph_engine_instance
    with _graph_lock:
        if _graph_engine_instance is None:
            _graph_engine_instance = EntityGraphEngine()
    return _graph_engine_instance
