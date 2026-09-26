"""
AHRAS Provenance Graph Builder
------------------------------
Ingests streaming OCSF telemetry events (process, network, file, auth, alert)
and dynamically materializes a heterogeneous, directional provenance graph
with sliding temporal window retention and entity correlation.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from provenance.models import (
    ProvenanceNodeType,
    ProvenanceEdgeType,
    ProvenanceNode,
    ProvenanceEdge,
)
from provenance.graph import ProvenanceGraph


class ProvenanceGraphBuilder:
    """
    Online ingestion and materialization engine for streaming provenance graphs.
    """
    def __init__(
        self,
        graph: Optional[ProvenanceGraph] = None,
        retention_window_s: float = 86400.0,  # 24 hours default retention
    ) -> None:
        self.graph = graph or ProvenanceGraph("live-provenance-stream")
        self.retention_window_s = retention_window_s
        self._event_count = 0

    def ingest_event(self, event: Dict[str, Any]) -> List[ProvenanceEdge]:
        """
        Parses an OCSF event and inserts corresponding nodes and causal edges into the graph.
        Returns the list of new edges created.
        """
        self._event_count += 1
        created_edges: List[ProvenanceEdge] = []

        ocsf_class = event.get("ocsf_class", event.get("class_name", "")).lower()
        ts = float(event.get("timestamp_epoch", event.get("time", time.time())))
        event_id = str(event.get("event_id", f"evt-{self._event_count}"))
        risk = float(event.get("risk_score", event.get("severity_id", 1) / 5.0))

        if "process" in ocsf_class or "command" in event:
            created_edges.extend(self._ingest_process_event(event, event_id, ts, risk))
        elif "network" in ocsf_class or "src_ip" in event or "src_endpoint" in event:
            created_edges.extend(self._ingest_network_event(event, event_id, ts, risk))
        elif "file" in ocsf_class or "file_name" in event or "file" in event:
            created_edges.extend(self._ingest_file_event(event, event_id, ts, risk))
        elif "auth" in ocsf_class or "user" in event:
            created_edges.extend(self._ingest_auth_event(event, event_id, ts, risk))
        else:
            # Generic / alert finding
            created_edges.extend(self._ingest_alert_event(event, event_id, ts, risk))

        return created_edges

    def _ingest_process_event(
        self, event: Dict[str, Any], event_id: str, ts: float, risk: float
    ) -> List[ProvenanceEdge]:
        edges: List[ProvenanceEdge] = []
        host_id = event.get("host_id", event.get("device", {}).get("hostname", "host-default"))
        pname = event.get("process_name", event.get("actor", {}).get("process", {}).get("name", "unknown_proc"))
        pid = event.get("pid", event.get("actor", {}).get("process", {}).get("pid", 1000))
        proc_node_id = f"{host_id}:proc:{pname}:{pid}"

        # 1. Host node
        self.graph.add_node(ProvenanceNode(
            node_id=host_id,
            node_type=ProvenanceNodeType.HOST,
            name=host_id,
            first_seen=ts,
            last_seen=ts,
            risk_score=risk * 0.5,
        ))

        # 2. Process node
        self.graph.add_node(ProvenanceNode(
            node_id=proc_node_id,
            node_type=ProvenanceNodeType.PROCESS,
            name=pname,
            properties={"cmd": event.get("command", ""), "pid": pid},
            first_seen=ts,
            last_seen=ts,
            risk_score=risk,
        ))

        # Edge: Host executes Process
        e1 = ProvenanceEdge(
            edge_id=str(uuid.uuid4()),
            source=host_id,
            target=proc_node_id,
            relation=ProvenanceEdgeType.EXECUTES,
            timestamp=ts,
            confidence=0.99,
            event_id=event_id,
        )
        self.graph.add_edge(e1)
        edges.append(e1)

        # 3. Parent process linkage if available
        parent_pname = event.get("parent_process_name", event.get("actor", {}).get("process", {}).get("parent", {}).get("name"))
        if parent_pname:
            parent_node_id = f"{host_id}:proc:{parent_pname}"
            self.graph.add_node(ProvenanceNode(
                node_id=parent_node_id,
                node_type=ProvenanceNodeType.PROCESS,
                name=parent_pname,
                first_seen=ts,
                last_seen=ts,
            ))
            e_parent = ProvenanceEdge(
                edge_id=str(uuid.uuid4()),
                source=parent_node_id,
                target=proc_node_id,
                relation=ProvenanceEdgeType.CREATES,
                timestamp=ts,
                confidence=0.95,
                event_id=event_id,
            )
            self.graph.add_edge(e_parent)
            edges.append(e_parent)

        return edges

    def _ingest_network_event(
        self, event: Dict[str, Any], event_id: str, ts: float, risk: float
    ) -> List[ProvenanceEdge]:
        edges: List[ProvenanceEdge] = []
        src_ip = event.get("src_ip", event.get("src_endpoint", {}).get("ip", "10.0.0.1"))
        dst_ip = event.get("dst_ip", event.get("dst_endpoint", {}).get("ip", "1.1.1.1"))
        dst_port = event.get("dst_port", event.get("dst_endpoint", {}).get("port", 80))

        # Source IP Node
        self.graph.add_node(ProvenanceNode(
            node_id=src_ip,
            node_type=ProvenanceNodeType.IP,
            name=src_ip,
            first_seen=ts,
            last_seen=ts,
            risk_score=risk * 0.6,
        ))

        # Destination IP Node
        self.graph.add_node(ProvenanceNode(
            node_id=dst_ip,
            node_type=ProvenanceNodeType.IP,
            name=dst_ip,
            first_seen=ts,
            last_seen=ts,
            risk_score=risk,
        ))

        # Edge: CONNECTS_TO or COMMUNICATES
        rel = ProvenanceEdgeType.CONNECTS_TO if risk > 0.4 else ProvenanceEdgeType.COMMUNICATES
        e = ProvenanceEdge(
            edge_id=str(uuid.uuid4()),
            source=src_ip,
            target=dst_ip,
            relation=rel,
            timestamp=ts,
            confidence=0.98,
            event_id=event_id,
            metadata={"port": dst_port, "protocol": event.get("protocol", "TCP")},
        )
        self.graph.add_edge(e)
        edges.append(e)

        return edges

    def _ingest_file_event(
        self, event: Dict[str, Any], event_id: str, ts: float, risk: float
    ) -> List[ProvenanceEdge]:
        edges: List[ProvenanceEdge] = []
        host_id = event.get("host_id", "host-default")
        file_path = event.get("file_path", event.get("file", {}).get("name", "/tmp/file.dat"))
        proc_name = event.get("process_name", "system")
        proc_node_id = f"{host_id}:proc:{proc_name}"
        file_node_id = f"{host_id}:file:{file_path}"

        self.graph.add_node(ProvenanceNode(
            node_id=file_node_id,
            node_type=ProvenanceNodeType.FILE,
            name=file_path,
            first_seen=ts,
            last_seen=ts,
            risk_score=risk,
        ))

        activity = event.get("activity_name", event.get("action", "write")).lower()
        rel = ProvenanceEdgeType.WRITES if "write" in activity or "create" in activity else ProvenanceEdgeType.READS

        e = ProvenanceEdge(
            edge_id=str(uuid.uuid4()),
            source=proc_node_id,
            target=file_node_id,
            relation=rel,
            timestamp=ts,
            confidence=0.95,
            event_id=event_id,
        )
        self.graph.add_edge(e)
        edges.append(e)

        return edges

    def _ingest_auth_event(
        self, event: Dict[str, Any], event_id: str, ts: float, risk: float
    ) -> List[ProvenanceEdge]:
        edges: List[ProvenanceEdge] = []
        user = event.get("user", event.get("user_name", "user-alice"))
        host_id = event.get("host_id", "host-default")
        user_node_id = f"user:{user}"

        self.graph.add_node(ProvenanceNode(
            node_id=user_node_id,
            node_type=ProvenanceNodeType.USER,
            name=user,
            first_seen=ts,
            last_seen=ts,
            risk_score=risk,
        ))

        e = ProvenanceEdge(
            edge_id=str(uuid.uuid4()),
            source=user_node_id,
            target=host_id,
            relation=ProvenanceEdgeType.AUTHENTICATES,
            timestamp=ts,
            confidence=0.99,
            event_id=event_id,
        )
        self.graph.add_edge(e)
        edges.append(e)

        return edges

    def _ingest_alert_event(
        self, event: Dict[str, Any], event_id: str, ts: float, risk: float
    ) -> List[ProvenanceEdge]:
        edges: List[ProvenanceEdge] = []
        alert_name = event.get("rule_name", event.get("alert_type", "HighRiskAlert"))
        target_entity = event.get("target_entity", event.get("host_id", "host-default"))
        alert_node_id = f"alert:{alert_name}:{event_id}"

        self.graph.add_node(ProvenanceNode(
            node_id=alert_node_id,
            node_type=ProvenanceNodeType.ALERT,
            name=alert_name,
            first_seen=ts,
            last_seen=ts,
            risk_score=max(0.85, risk),
        ))

        e = ProvenanceEdge(
            edge_id=str(uuid.uuid4()),
            source=target_entity,
            target=alert_node_id,
            relation=ProvenanceEdgeType.TRIGGERS,
            timestamp=ts,
            confidence=0.99,
            event_id=event_id,
        )
        self.graph.add_edge(e)
        edges.append(e)

        return edges
