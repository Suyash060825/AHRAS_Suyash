"""
AHRAS Subgraph Extractor
------------------------
Extracts focused k-hop causal provenance neighborhoods around alert seeds,
calculating upstream root-cause lineage and downstream blast radius.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from provenance.models import ProvenanceNode, ProvenanceEdge
from provenance.graph import ProvenanceGraph


class SubgraphExtractor:
    """
    Extracts time-bounded, directional k-hop subgraphs centered on seed entities.
    """
    def __init__(self, full_graph: ProvenanceGraph) -> None:
        self.full_graph = full_graph

    def extract_causal_neighborhood(
        self,
        seed_node_ids: List[str],
        max_backward_hops: int = 3,
        max_forward_hops: int = 2,
        time_window_s: Optional[float] = None,
        subgraph_id: str = "extracted-subgraph",
    ) -> ProvenanceGraph:
        """
        Extracts upstream causal predecessors (backward) and downstream consequence (forward)
        nodes and edges within specified hop radii and temporal window.
        """
        subgraph = ProvenanceGraph(graph_id=subgraph_id)
        visited_nodes: Set[str] = set()
        included_edges: Dict[str, ProvenanceEdge] = {}

        # Determine reference time from seeds if time window is specified
        seed_timestamps = [
            self.full_graph.nodes[nid].last_seen
            for nid in seed_node_ids
            if nid in self.full_graph.nodes
        ]
        ref_time = max(seed_timestamps) if seed_timestamps else None

        # 1. Add all seed nodes
        for nid in seed_node_ids:
            if nid in self.full_graph.nodes:
                subgraph.add_node(self.full_graph.nodes[nid])
                visited_nodes.add(nid)

        # 2. Backward Causal Search (Ancestors / Root Cause)
        queue_back = deque([(nid, 0) for nid in seed_node_ids if nid in self.full_graph.nodes])
        while queue_back:
            curr_id, depth = queue_back.popleft()
            if depth >= max_backward_hops:
                continue

            for edge in self.full_graph.get_incoming_edges(curr_id):
                # Temporal filtering
                if time_window_s and ref_time is not None:
                    if abs(edge.timestamp - ref_time) > time_window_s:
                        continue

                included_edges[edge.edge_id] = edge
                parent_id = edge.source
                if parent_id in self.full_graph.nodes and parent_id not in visited_nodes:
                    visited_nodes.add(parent_id)
                    subgraph.add_node(self.full_graph.nodes[parent_id])
                    queue_back.append((parent_id, depth + 1))

        # 3. Forward Impact Search (Descendants / Blast Radius)
        queue_fwd = deque([(nid, 0) for nid in seed_node_ids if nid in self.full_graph.nodes])
        while queue_fwd:
            curr_id, depth = queue_fwd.popleft()
            if depth >= max_forward_hops:
                continue

            for edge in self.full_graph.get_outgoing_edges(curr_id):
                if time_window_s and ref_time is not None:
                    if abs(edge.timestamp - ref_time) > time_window_s:
                        continue

                included_edges[edge.edge_id] = edge
                child_id = edge.target
                if child_id in self.full_graph.nodes and child_id not in visited_nodes:
                    visited_nodes.add(child_id)
                    subgraph.add_node(self.full_graph.nodes[child_id])
                    queue_fwd.append((child_id, depth + 1))

        # 4. Insert all collected edges
        for edge in included_edges.values():
            subgraph.add_edge(edge)

        return subgraph
