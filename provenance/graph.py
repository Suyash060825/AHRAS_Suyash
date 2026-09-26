from __future__ import annotations
"""
AHRAS Module / Provenance — Heterogeneous Provenance Graph Engine
-----------------------------------------------------------------
Maintains a directional, temporal, heterogeneous provenance DAG:
  - Nodes: IP, HOST, USER, ACCOUNT, PROCESS, FILE, DOMAIN, SOCKET, ASSET, ALERT, TECHNIQUE, INCIDENT.
  - Edges: CONNECTS_TO, EXECUTES, READS, WRITES, CREATES, AUTHENTICATES, RESOLVES, COMMUNICATES, TARGETS, USES_TECHNIQUE, TRIGGERS, PART_OF.
Supports forward/backward causal traversals, path enumeration, and temporal windowing.
"""

from collections import defaultdict, deque
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from provenance.models import (
    ProvenanceNode, ProvenanceEdge, ProvenanceNodeType, ProvenanceEdgeType
)


class ProvenanceGraph:
    """
    Directional, attributed temporal property graph storing heterogeneous forensic provenance.
    """

    def __init__(self, graph_id: str = "prov-graph-01"):
        self.graph_id = graph_id
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: Dict[str, ProvenanceEdge] = {}
        self._adj_out: Dict[str, List[ProvenanceEdge]] = defaultdict(list)
        self._adj_in: Dict[str, List[ProvenanceEdge]] = defaultdict(list)

    def add_node(self, node: ProvenanceNode) -> None:
        """Adds or updates a provenance node."""
        if node.node_id in self.nodes:
            existing = self.nodes[node.node_id]
            existing.last_seen = max(existing.last_seen, node.last_seen)
            existing.risk_score = max(existing.risk_score, node.risk_score)
            existing.properties.update(node.properties)
        else:
            self.nodes[node.node_id] = copy.deepcopy(node)

    def add_edge(self, edge: ProvenanceEdge) -> None:
        """Adds an edge and registers adjacency. Ensures source and target exist."""
        # Auto-create fallback placeholder nodes if not present
        if edge.source not in self.nodes:
            self.add_node(ProvenanceNode(
                node_id=edge.source,
                node_type=ProvenanceNodeType.HOST if "." not in edge.source else ProvenanceNodeType.IP,
                name=edge.source,
                first_seen=edge.timestamp,
                last_seen=edge.timestamp,
            ))
        if edge.target not in self.nodes:
            self.add_node(ProvenanceNode(
                node_id=edge.target,
                node_type=ProvenanceNodeType.HOST if "." not in edge.target else ProvenanceNodeType.IP,
                name=edge.target,
                first_seen=edge.timestamp,
                last_seen=edge.timestamp,
            ))

        self.edges[edge.edge_id] = copy.deepcopy(edge)
        self._adj_out[edge.source].append(edge)
        self._adj_in[edge.target].append(edge)

        # Update node timestamp bounds
        self.nodes[edge.source].last_seen = max(self.nodes[edge.source].last_seen, edge.timestamp)
        self.nodes[edge.target].last_seen = max(self.nodes[edge.target].last_seen, edge.timestamp)

    def get_node(self, node_id: str) -> Optional[ProvenanceNode]:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[ProvenanceEdge]:
        return self.edges.get(edge_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def get_outgoing_edges(self, node_id: str) -> List[ProvenanceEdge]:
        return self._adj_out.get(node_id, [])

    def get_incoming_edges(self, node_id: str) -> List[ProvenanceEdge]:
        return self._adj_in.get(node_id, [])

    def get_neighbors(self, node_id: str) -> Set[str]:
        nbrs = set()
        for e in self.get_outgoing_edges(node_id):
            nbrs.add(e.target)
        for e in self.get_incoming_edges(node_id):
            nbrs.add(e.source)
        return nbrs

    # ── Graph Traversal & Reachability ────────────────────────────────────────

    def forward_reachability(self, start_node: str, max_depth: int = 10) -> Set[str]:
        """Finds all nodes reachable downstream from start_node via outgoing edges."""
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(start_node, 0)])

        while queue:
            curr, depth = queue.popleft()
            if curr in visited or depth > max_depth:
                continue
            visited.add(curr)

            for e in self.get_outgoing_edges(curr):
                if e.target not in visited:
                    queue.append((e.target, depth + 1))

        return visited

    def backward_causal_path(self, target_node: str, max_depth: int = 10) -> List[List[ProvenanceEdge]]:
        """
        Traces back causal antecedents from target_node upstream to identify root causes.
        Returns a list of edge paths leading into target_node.
        """
        paths: List[List[ProvenanceEdge]] = []

        def dfs(curr: str, current_path: List[ProvenanceEdge], depth: int, visited: Set[str]):
            if depth >= max_depth:
                if current_path:
                    paths.append(list(reversed(current_path)))
                return

            incoming = self.get_incoming_edges(curr)
            # Filter incoming edges to preserve temporal ordering (earlier than current)
            if current_path:
                incoming = [e for e in incoming if e.timestamp <= current_path[-1].timestamp]

            if not incoming:
                if current_path:
                    paths.append(list(reversed(current_path)))
                return

            for e in incoming:
                if e.source in visited:
                    continue
                visited.add(e.source)
                current_path.append(e)
                dfs(e.source, current_path, depth + 1, visited)
                current_path.pop()
                visited.remove(e.source)

        dfs(target_node, [], 0, {target_node})
        return paths

    def find_all_simple_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 8,
    ) -> List[List[ProvenanceEdge]]:
        """Finds all directed paths from source to target respecting temporal forward order."""
        if source not in self.nodes or target not in self.nodes:
            return []

        all_paths: List[List[ProvenanceEdge]] = []

        def dfs(curr: str, path: List[ProvenanceEdge], visited: Set[str], last_ts: float):
            if curr == target:
                all_paths.append(list(path))
                return
            if len(path) >= max_depth:
                return

            for edge in self.get_outgoing_edges(curr):
                # Ensure causal temporal consistency
                if edge.timestamp >= last_ts and edge.target not in visited:
                    visited.add(edge.target)
                    path.append(edge)
                    dfs(edge.target, path, visited, edge.timestamp)
                    path.pop()
                    visited.remove(edge.target)

        dfs(source, [], {source}, 0.0)
        return all_paths

    def filter_by_time_window(self, start_ts: float, end_ts: float) -> ProvenanceGraph:
        """Returns an induced subgraph containing only edges within [start_ts, end_ts]."""
        sub = ProvenanceGraph(graph_id=f"{self.graph_id}-window")
        for e in self.edges.values():
            if start_ts <= e.timestamp <= end_ts:
                if not sub.has_node(e.source):
                    sub.add_node(self.nodes[e.source])
                if not sub.has_node(e.target):
                    sub.add_node(self.nodes[e.target])
                sub.add_edge(e)
        return sub

    def get_subgraph(self, node_ids: Set[str]) -> ProvenanceGraph:
        """Extracts the induced subgraph formed by node_ids."""
        sub = ProvenanceGraph(graph_id=f"{self.graph_id}-sub")
        for nid in node_ids:
            if nid in self.nodes:
                sub.add_node(self.nodes[nid])
        for e in self.edges.values():
            if e.source in node_ids and e.target in node_ids:
                sub.add_edge(e)
        return sub

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges.values()],
        }
