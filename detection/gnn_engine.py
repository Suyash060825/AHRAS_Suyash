from __future__ import annotations
"""
AHRAS Module 6 — Entity Graph & Lateral Movement Detector
---------------------------------------------------------
Tracks entity relationship topologies across multi-hop user, host, process, and network nodes.

Attack Pattern Caught:
  Catches multi-stage lateral movement and privilege escalation attack chains where each
  individual step looks normal in isolation (e.g. 1 SSH login, 1 process launch),
  but the graph path across nodes forms an anomalous multi-hop intrusion chain.

Graph Node Types:
  - Host (IP / Hostname)
  - User (Username / Identity)
  - Process (Process Name / PID)
  - Resource (File Path / Cloud API Action)
"""

import math
import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GraphPathAnomaly:
    """Output from scoring multi-hop path anomaly."""
    source_entity:    str
    target_entity:    str
    path_length:      int
    path_nodes:       list[str]
    path_risk_score:  float
    is_lateral_movement: bool
    mitre_techniques: list[str] = field(default_factory=list)


class EntityGraphEngine:
    """
    Maintains a multi-hop entity graph and detects anomalous lateral movement paths.
    Thread-safe via RLock.
    """

    def __init__(self, max_hops: int = 5):
        # Graph adjacency: node -> set of (target_node, relation_type, timestamp)
        self._graph: Dict[str, Set[Tuple[str, str, float]]] = defaultdict(set)
        # Node access frequencies: node -> count
        self._node_freq: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._max_hops = max_hops

    def add_event_edge(self, src_node: str, dst_node: str, relation: str, ts: float = None) -> None:
        """Adds a directed edge between two entities in the security graph."""
        if not src_node or not dst_node or src_node == dst_node:
            return
        if ts is None:
            ts = time.time()

        with self._lock:
            self._graph[src_node].add((dst_node, relation, ts))
            self._node_freq[src_node] += 1
            self._node_freq[dst_node] += 1

    def analyze_lateral_movement(self, src_node: str, dst_node: str) -> GraphPathAnomaly:
        """
        Calculates shortest path and evaluates lateral movement path anomaly score.
        """
        with self._lock:
            path = self._find_shortest_path(src_node, dst_node)
            src_freq = self._node_freq[src_node]
            dst_freq = self._node_freq[dst_node]

        if not path:
            return GraphPathAnomaly(
                source_entity=src_node, target_entity=dst_node, path_length=0,
                path_nodes=[], path_risk_score=0.0, is_lateral_movement=False
            )

        path_len = len(path) - 1
        # Rarity score: path containing low-frequency nodes is more anomalous
        rarity_score = 1.0 / math.sqrt(max(dst_freq, 1))
        path_risk = min((path_len * 0.25) + (rarity_score * 0.5), 1.0)
        path_risk = round(float(path_risk), 4)

        is_anom = path_len >= 3 or path_risk >= 0.70
        mitre = []
        if is_anom:
            mitre.append("T1021")  # Remote Services / Lateral Movement
            mitre.append("T1078")  # Valid Accounts

        return GraphPathAnomaly(
            source_entity=src_node,
            target_entity=dst_node,
            path_length=path_len,
            path_nodes=path,
            path_risk_score=path_risk,
            is_lateral_movement=is_anom,
            mitre_techniques=mitre,
        )

    def _find_shortest_path(self, start: str, end: str) -> List[str]:
        """BFS shortest path finder across entity graph."""
        if start not in self._graph:
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

            for neighbor, _, _ in self._graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return []


# Singleton
_graph_engine_instance: Optional[EntityGraphEngine] = None
_graph_lock = threading.Lock()


def get_entity_graph_engine() -> EntityGraphEngine:
    global _graph_engine_instance
    with _graph_lock:
        if _graph_engine_instance is None:
            _graph_engine_instance = EntityGraphEngine()
    return _graph_engine_instance
