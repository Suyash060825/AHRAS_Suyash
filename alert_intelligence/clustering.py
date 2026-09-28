from __future__ import annotations
"""
AHRAS Alert Clustering Engine
==============================
Correlates discrete security alerts into cohesive, multi-stage IncidentClusters.

Clustering Mechanisms:
1. Spatial Proximity: Direct entity key match or shared network subnet.
2. Temporal Window: Alerts within temporal correlation threshold Δt (default 300s).
3. Kill-Chain Progression: Tracks MITRE ATT&CK tactics to identify attack advancement.
4. Dynamic Merging: Automatically merges separate clusters when lateral movement
   or pivot actions bridge previously distinct entities.
"""

import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from alert_intelligence.models import IncidentCluster, RawAlert

# MITRE Enterprise Tactic Progression Hierarchy (Sequential Kill Chain Order)
TACTIC_PROGRESSION_ORDER = [
    "TA0043", # Reconnaissance
    "TA0042", # Resource Development
    "TA0001", # Initial Access
    "TA0002", # Execution
    "TA0003", # Persistence
    "TA0004", # Privilege Escalation
    "TA0005", # Defense Evasion
    "TA0006", # Credential Access
    "TA0007", # Discovery
    "TA0008", # Lateral Movement
    "TA0009", # Collection
    "TA0011", # Command and Control
    "TA0010", # Exfiltration
    "TA0040", # Impact
]

TACTIC_NAMES = {
    "TA0043": "Reconnaissance",
    "TA0042": "Resource Development",
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0011": "Command & Control",
    "TA0010": "Exfiltration",
    "TA0040": "Impact",
}


class AlertClusteringEngine:
    """
    Online spatio-temporal alert clustering engine.
    Ingests deduplicated RawAlerts and maintains an active state of open IncidentClusters.
    """

    def __init__(self, correlation_window_seconds: float = 300.0, max_inactive_seconds: float = 1800.0):
        self.correlation_window_seconds = correlation_window_seconds
        self.max_inactive_seconds = max_inactive_seconds
        self._lock = threading.Lock()
        
        # cluster_id -> IncidentCluster
        self._active_clusters: Dict[str, IncidentCluster] = {}
        # entity_key -> Set[cluster_id] for rapid entity lookup
        self._entity_index: Dict[str, Set[str]] = {}

    def correlate(self, alert: RawAlert) -> IncidentCluster:
        """
        Assigns an alert to an existing matching cluster or initializes a new one.
        Handles dynamic multi-cluster merging if the alert bridges multiple entities.
        """
        now = alert.timestamp if alert.timestamp > 0 else time.time()
        
        with self._lock:
            self._prune_inactive(now)
            
            candidate_cluster_ids = self._find_matching_clusters(alert, now)
            
            if not candidate_cluster_ids:
                # No match found -> spawn new cluster
                new_cluster = self._create_new_cluster(alert)
                self._active_clusters[new_cluster.cluster_id] = new_cluster
                self._index_cluster(new_cluster)
                return new_cluster
            
            if len(candidate_cluster_ids) == 1:
                # Single matching cluster -> append alert
                target_id = next(iter(candidate_cluster_ids))
                cluster = self._active_clusters[target_id]
                cluster.add_alert(alert)
                self._update_cluster_title_and_progression(cluster)
                self._index_cluster(cluster)
                return cluster
            
            # Multiple matching clusters -> Bridge entity! Merge candidate clusters into one.
            merged_cluster = self._merge_clusters(candidate_cluster_ids, alert)
            return merged_cluster

    def _find_matching_clusters(self, alert: RawAlert, current_time: float) -> Set[str]:
        """Finds all active clusters that match the alert by entity or temporal proximity."""
        matching = set()
        
        # 1. Direct entity match
        if alert.entity_key in self._entity_index:
            for cid in self._entity_index[alert.entity_key]:
                cluster = self._active_clusters.get(cid)
                if cluster and (current_time - cluster.last_update_time) <= self.correlation_window_seconds:
                    matching.add(cid)
                    
        # 2. Check metadata related entities (e.g. src_ip, dest_ip, parent_process)
        related_entities = alert.metadata.get("related_entities", [])
        if isinstance(related_entities, list):
            for rel in related_entities:
                if rel in self._entity_index:
                    for cid in self._entity_index[rel]:
                        cluster = self._active_clusters.get(cid)
                        if cluster and (current_time - cluster.last_update_time) <= self.correlation_window_seconds:
                            matching.add(cid)
                            
        return matching

    def _create_new_cluster(self, alert: RawAlert) -> IncidentCluster:
        """Initializes a new IncidentCluster from a solitary alert."""
        tactic_name = TACTIC_NAMES.get(alert.mitre_tactic or "", alert.mitre_tactic or "Anomaly")
        title = f"{tactic_name} on {alert.entity_key}"
        
        cluster = IncidentCluster(
            title=title,
            entity_keys=[alert.entity_key],
            primary_entity=alert.entity_key,
            start_time=alert.timestamp,
            last_update_time=alert.timestamp,
            alerts=[alert],
            tactics_present=[alert.mitre_tactic] if alert.mitre_tactic else [],
            techniques_present=[alert.mitre_technique] if alert.mitre_technique else [],
            evidence_ids=list(alert.evidence_ids),
        )
        self._update_cluster_title_and_progression(cluster)
        return cluster

    def _merge_clusters(self, cluster_ids: Set[str], bridging_alert: RawAlert) -> IncidentCluster:
        """Merges multiple clusters into the oldest surviving cluster."""
        sorted_ids = sorted(
            cluster_ids,
            key=lambda cid: self._active_clusters[cid].start_time
        )
        primary_id = sorted_ids[0]
        primary_cluster = self._active_clusters[primary_id]
        
        for other_id in sorted_ids[1:]:
            other_cluster = self._active_clusters.pop(other_id, None)
            if not other_cluster:
                continue
            # Merge alerts
            for a in other_cluster.alerts:
                primary_cluster.add_alert(a)
            # Remove from index
            for ent in other_cluster.entity_keys:
                if ent in self._entity_index:
                    self._entity_index[ent].discard(other_id)
        
        # Add bridging alert
        primary_cluster.add_alert(bridging_alert)
        self._update_cluster_title_and_progression(primary_cluster)
        self._index_cluster(primary_cluster)
        return primary_cluster

    def _update_cluster_title_and_progression(self, cluster: IncidentCluster) -> None:
        """Updates progression score and generates an informative incident title."""
        # Calculate kill-chain progression score
        tactics = cluster.tactics_present
        progression_indices = [
            TACTIC_PROGRESSION_ORDER.index(t)
            for t in tactics
            if t in TACTIC_PROGRESSION_ORDER
        ]
        
        if progression_indices:
            span = max(progression_indices) - min(progression_indices) + 1
            progression_score = min(1.0, (len(set(progression_indices)) * 0.2) + (span / len(TACTIC_PROGRESSION_ORDER)) * 0.5)
        else:
            progression_score = min(1.0, len(cluster.alerts) * 0.05)
            
        cluster.progression_score = round(progression_score, 4)
        
        # Format title based on kill chain span
        ordered_tactics = [
            TACTIC_NAMES.get(t, t)
            for t in TACTIC_PROGRESSION_ORDER
            if t in tactics
        ]
        
        entity_desc = cluster.primary_entity
        if len(cluster.entity_keys) > 1:
            entity_desc += f" (+{len(cluster.entity_keys)-1} entities)"
            
        if len(ordered_tactics) >= 2:
            cluster.title = f"Multi-Stage Attack: {' -> '.join(ordered_tactics[:3])} on {entity_desc}"
        elif ordered_tactics:
            cluster.title = f"{ordered_tactics[0]} activity on {entity_desc}"
        else:
            cluster.title = f"Correlated Anomaly Campaign on {entity_desc}"

    def _index_cluster(self, cluster: IncidentCluster) -> None:
        """Updates the entity-to-cluster index."""
        for ent in cluster.entity_keys:
            if ent not in self._entity_index:
                self._entity_index[ent] = set()
            self._entity_index[ent].add(cluster.cluster_id)

    def _prune_inactive(self, current_time: float) -> None:
        """Prunes clusters that have not received an alert within max_inactive_seconds."""
        cutoff = current_time - self.max_inactive_seconds
        inactive = [
            cid for cid, c in self._active_clusters.items()
            if c.last_update_time < cutoff
        ]
        for cid in inactive:
            c = self._active_clusters.pop(cid)
            for ent in c.entity_keys:
                if ent in self._entity_index:
                    self._entity_index[ent].discard(cid)

    def get_cluster(self, cluster_id: str) -> Optional[IncidentCluster]:
        with self._lock:
            return self._active_clusters.get(cluster_id)

    def get_all_clusters(self) -> List[IncidentCluster]:
        with self._lock:
            return list(self._active_clusters.values())

    @property
    def active_cluster_count(self) -> int:
        with self._lock:
            return len(self._active_clusters)
